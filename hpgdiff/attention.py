from inspect import isfunction
import math
import os
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from hpgdiff.util import  checkpoint, normalization
from .fp16_util import convert_module_to_f16_transformer


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):

    for p in module.parameters():
        p.detach().zero_()
    return module


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = normalization(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)


        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)


        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)


        attn = sim.softmax(dim=-1)
        self.last_attn = attn.detach()

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dtype, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.dtype = dtype
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):

        x = self.attn1(self.norm1(x.float()).type(self.dtype)) + x
        cross_attn_out = self.attn2(self.norm2(x.float()).type(self.dtype), context=context)
        self.last_cross_attn_output = cross_attn_out.detach()
        self.last_cross_attn_weights = self.attn2.last_attn.detach()
        x = cross_attn_out + x
        x = self.ff(self.norm3(x.float()).type(self.dtype)) + x
        return x


class SpatialTransformer(nn.Module):

    def __init__(self, in_channels, n_heads, d_head, use_fp16, dtype,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()


        if isinstance(context_dim, str):
            context_dim = int(context_dim)

        self.in_channels = in_channels
        inner_dim = n_heads * d_head

        self.use_fp16 = use_fp16
        self.dtype = dtype

        self.norm = normalization(in_channels)
        self.proj_in = nn.Conv2d(in_channels,
                                inner_dim,
                                kernel_size=1,
                                stride=1,
                                padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, self.dtype, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                            in_channels,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0))
        if self.use_fp16:
            self.convert_to_fp16()


    _psl_save_counter = 0

    @staticmethod
    def _save_map(arr, path):
        import numpy as np
        from PIL import Image

        lo, hi = np.percentile(arr, [1, 99])
        arr = np.clip(arr, lo, hi)
        if arr.max() > arr.min():
            arr = (arr - arr.min()) / (arr.max() - arr.min())
        rgb = np.zeros((*arr.shape, 3), dtype=np.float32)
        rgb[..., 0] = np.clip(1.7 * arr, 0, 1)
        rgb[..., 1] = np.clip(1.7 * (1.0 - np.abs(arr - 0.55) * 2.0), 0, 1)
        rgb[..., 2] = np.clip(1.5 * (1.0 - arr), 0, 1)
        image = Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")
        if os.environ.get("SAVE_PSL_ATTN_ROTATE_CW", "1") == "1":
            image = image.transpose(Image.Transpose.ROTATE_270)
        image.save(path)

    @staticmethod
    def _save_branch_cross_attention_output(cross_attn_out, cross_attn_weights, h, w, branch):

        if os.environ.get("SAVE_PSL_ATTN_MAPS", "0") != "1":
            return

        current_t = int(float(os.environ.get("SAVE_PSL_ATTN_CURRENT_T", "1000000000")))
        t_max = int(float(os.environ.get("SAVE_PSL_ATTN_T_MAX", "50")))
        if current_t > t_max:
            return

        max_saves = int(os.environ.get("SAVE_PSL_ATTN_MAX", "64"))
        if SpatialTransformer._psl_save_counter >= max_saves:
            return

        import numpy as np

        save_dir = os.environ.get("SAVE_PSL_ATTN_DIR", "./feature_maps_psl_attn")
        os.makedirs(save_dir, exist_ok=True)
        batch_index = int(os.environ.get("SAVE_PSL_ATTN_BATCH_INDEX", "0"))

        feature = rearrange(cross_attn_out.detach().float().cpu(), "b (h w) c -> b c h w", h=h, w=w)
        if batch_index < 0 or batch_index >= feature.shape[0]:
            return

        sample_tag = os.environ.get("SAVE_PSL_ATTN_SAMPLE_INDEX", "sample")
        activation = feature.abs().mean(dim=1)[batch_index].numpy()
        prefix = f"{branch}_cross_attn"
        raw_path = os.path.join(
            save_dir,
            f"{prefix}_output_{sample_tag}_t{current_t:04d}_{SpatialTransformer._psl_save_counter:04d}_res_{h}x{w}.npy",
        )
        png_path = os.path.join(
            save_dir,
            f"{prefix}_output_{sample_tag}_t{current_t:04d}_{SpatialTransformer._psl_save_counter:04d}_res_{h}x{w}.png",
        )

        np.save(raw_path, activation)
        SpatialTransformer._save_map(activation, png_path)

        if cross_attn_weights is not None:
            b = cross_attn_out.shape[0]
            heads = cross_attn_weights.shape[0] // b
            attn = cross_attn_weights.detach().float().cpu().view(b, heads, h * w, h * w)
            context_attention = attn.mean(dim=(1, 2))[batch_index].view(h, w).numpy()
            weights_raw_path = os.path.join(
                save_dir,
                f"{prefix}_weights_{sample_tag}_t{current_t:04d}_{SpatialTransformer._psl_save_counter:04d}_res_{h}x{w}.npy",
            )
            weights_png_path = os.path.join(
                save_dir,
                f"{prefix}_weights_{sample_tag}_t{current_t:04d}_{SpatialTransformer._psl_save_counter:04d}_res_{h}x{w}.png",
            )
            np.save(weights_raw_path, context_attention)
            SpatialTransformer._save_map(context_attention, weights_png_path)

        SpatialTransformer._psl_save_counter += 1

    def convert_to_fp16(self):

        self.proj_in.apply(convert_module_to_f16_transformer)
        self.proj_out.apply(convert_module_to_f16_transformer)
        self.transformer_blocks.apply(convert_module_to_f16_transformer)

    def forward(self, x, context_1=None, context_2=None, context_3=None):


        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)


        context_source = None
        if h == 32 and w == 32 and context_1 is not None:
            context = context_1
            context_source = "u"
        elif h == 16 and w == 16 and context_2 is not None:
            context = context_2
            context_source = "psl"
        elif h == 8 and w == 8 and context_3 is not None:
            context = context_3
            context_source = "energy"

        if context.shape[1] != x.shape[1]:

            context_conv = nn.Conv2d(
                context.shape[1], x.shape[1], kernel_size=1,
                stride=1, padding=0
            ).to(context.device).to(context.dtype)
            context = context_conv(context)


        x = rearrange(x, 'b c h w -> b (h w) c')
        context = rearrange(context, 'b c h w -> b (h w) c')


        target_branch = os.environ.get("SAVE_ATTN_BRANCH", "psl")
        for block in self.transformer_blocks:
            x = block(x, context=context)
            if context_source == target_branch and hasattr(block, "last_cross_attn_output"):
                self._save_branch_cross_attention_output(
                    block.last_cross_attn_output,
                    getattr(block, "last_cross_attn_weights", None),
                    h,
                    w,
                    target_branch,
                )


        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in
