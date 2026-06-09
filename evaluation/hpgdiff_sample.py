import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch as th
import torch.distributed as dist

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from cons_input_datasets import load_data
from hpgdiff import dist_util, logger
from hpgdiff.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    data = load_data(
        test_level=args.test_level
    )

    def model_fn(x, t, **model_kwargs):
        if os.environ.get("SAVE_PSL_ATTN_ARMED", "0") == "1":
            os.environ["SAVE_PSL_ATTN_CURRENT_T"] = str(int(t[0].detach().cpu().item()))
        return model(x, t, **model_kwargs)

    logger.log("sampling...")
    all_images = []
    while len(all_images) * args.batch_size < args.num_samples:
        batch_start_index = len(all_images) * args.batch_size
        model_kwargs = {}
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        all_input_cons = []
        all_input_raw_loads = []
        all_input_raw_BCs = []
        energy_mask = []
        psl_mask = []
        disp_mask = []

        for i in range(args.batch_size):
            input_cons, input_raw_loads, input_raw_BCs, mask_1, mask_2, mask_3, _ = next(data)
            all_input_cons.append(input_cons)
            all_input_raw_loads.append(input_raw_loads)
            all_input_raw_BCs.append(input_raw_BCs)
            energy_mask.append(mask_1)
            psl_mask.append(mask_2)
            disp_mask.append(mask_3)
        input_cons = th.cat(all_input_cons, dim=0).cuda()
        input_raw_loads = th.cat(all_input_raw_loads, dim=0).cuda()
        input_raw_BCs = th.cat(all_input_raw_BCs, dim=0).cuda()
        energy_mask = th.cat(energy_mask, dim=0).cuda()
        psl_mask = th.cat(psl_mask, dim=0).cuda()
        disp_mask = th.cat(disp_mask, dim=0).cuda()
        all_masks = th.cat((energy_mask, psl_mask, disp_mask), dim = 1)
        model_kwargs["context"] = all_masks

        target_index = int(args.psl_attn_sample_index)
        should_save_psl_attn = (
            target_index >= 0
            and batch_start_index <= target_index < batch_start_index + args.batch_size
        )
        if should_save_psl_attn:
            local_index = target_index - batch_start_index
            os.environ["SAVE_PSL_ATTN_ARMED"] = "1"
            os.environ["SAVE_PSL_ATTN_MAPS"] = "1"
            os.environ["SAVE_PSL_ATTN_BATCH_INDEX"] = str(local_index)
            os.environ["SAVE_PSL_ATTN_SAMPLE_INDEX"] = str(target_index)
            os.environ["SAVE_PSL_ATTN_DIR"] = args.psl_attn_dir
            os.environ["SAVE_PSL_ATTN_MAX"] = str(args.psl_attn_max)
            os.environ["SAVE_PSL_ATTN_T_MAX"] = str(args.psl_attn_t_max)
            os.environ["SAVE_ATTN_BRANCH"] = args.attn_branch
            logger.log(
                f"saving {args.attn_branch} cross-attention maps for sample {target_index} "
                f"(batch local index {local_index}, t <= {args.psl_attn_t_max}) "
                f"to {args.psl_attn_dir}"
            )
        else:
            os.environ["SAVE_PSL_ATTN_ARMED"] = "0"
            os.environ["SAVE_PSL_ATTN_MAPS"] = "0"

        sample = sample_fn(
            model_fn,
            (args.batch_size, 1, args.image_size, args.image_size),
            input_cons,
            input_raw_loads,
            input_raw_BCs,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
        )
        sample = (sample * 255).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        logger.log(f"created {len(all_images)} batch samples")
        th.cuda.empty_cache()

        if should_save_psl_attn:
            os.environ["SAVE_PSL_ATTN_ARMED"] = "0"
            os.environ["SAVE_PSL_ATTN_MAPS"] = "0"

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=20,
        batch_size=1,
        use_ddim=False,
        model_path="",
        test_level="",
        psl_attn_sample_index=-1,
        psl_attn_dir="feature_maps_psl_attn",
        psl_attn_max=64,
        psl_attn_t_max=50,
        attn_branch="psl",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()
