import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
from PIL import Image
from tqdm.auto import tqdm


from peft import LoraConfig, get_peft_model
from bitsandbytes.optim import AdamW8bit
from accelerate import Accelerator


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from hpgdiff.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)


class TrainingConfig:

    pretrained_model_path = "../checkpoints/local_loss_0.01/model120000.pt"
    dataset_dir = "./LoRA_data"
    output_dir = "./lora_checkpoints"

    image_size = 64
    original_height = 20
    original_width = 60
    train_batch_size = 40
    num_epochs = 20
    learning_rate = 2e-5


config = TrainingConfig()


class LoRATopologyDataset(Dataset):
    def __init__(self, data_dir, image_size):
        self.data_dir = data_dir
        self.image_size = image_size


        self.clean_images_dir = os.path.join(data_dir, 'clean_images_60x20')
        self.pfs_dir = os.path.join(data_dir, 'pfs_60x20')
        self.psls_dir = os.path.join(data_dir, 'psls_60x20')
        self.disp_masks_dir = os.path.join(data_dir, 'disp_masks_60x20')
        self.loads_dir = os.path.join(data_dir, 'load_fields_60x20')


        self.filenames = []
        for filename in os.listdir(self.clean_images_dir):
            if filename.endswith('.png'):

                sample_id = filename.replace('sample_', '').replace('.png', '')
                self.filenames.append(sample_id)


    def __len__(self):
        return len(self.filenames)

    def _pad_to_canvas(self, data, target_size=64):

        h, w = data.shape[:2]


        pad_h = (target_size - h) // 2
        pad_w = (target_size - w) // 2


        if data.ndim == 3:
            padded = np.zeros((target_size, target_size, data.shape[2]), dtype=data.dtype)
            padded[pad_h:pad_h+h, pad_w:pad_w+w, :] = data
        elif data.ndim == 2:
            padded = np.zeros((target_size, target_size), dtype=data.dtype)
            padded[pad_h:pad_h+h, pad_w:pad_w+w] = data

        return padded

    def _extract_isolated_materials(self, image, max_material_size=5):

        from scipy.ndimage import label


        if image.max() > 1:
            image = image / 255.0
        binary_img = (image > 0.5).astype(np.uint8)


        inverted_img = 1 - binary_img
        labeled_material, num_material = label(inverted_img)

        material_sizes = np.bincount(labeled_material.ravel())


        isolated_materials = np.where((material_sizes > 0) &
                                     (material_sizes < max_material_size))[0]


        isolated_mask = np.zeros_like(binary_img)
        small_materials = np.isin(labeled_material, isolated_materials)
        isolated_mask[small_materials] = 1


        isolated_mask = 1 - isolated_mask
        isolated_mask = isolated_mask * 255.0

        return isolated_mask

    def __getitem__(self, idx):
        sample_id = self.filenames[idx]

        try:

            target_img_path = os.path.join(self.clean_images_dir, f'sample_{sample_id}.png')
            target_img = Image.open(target_img_path).convert("L")
            target_array = np.array(target_img)


            target_padded = self._pad_to_canvas(target_array, self.image_size)


            image_plus = self._extract_isolated_materials(target_padded)


            target_tensor = torch.from_numpy(target_padded).float() / 127.5 - 1.0
            target_tensor = target_tensor.unsqueeze(0)

            image_plus_tensor = torch.from_numpy(image_plus).float() / 127.5 - 1.0
            image_plus_tensor = image_plus_tensor.unsqueeze(0)


            pf_path = os.path.join(self.pfs_dir, f'cons_pf_array_{sample_id}.npy')
            pf_data = np.load(pf_path)


            psl_path = os.path.join(self.psls_dir, f'sample_{sample_id}.png')
            psl_img = Image.open(psl_path).convert("RGB")
            psl_array = np.array(psl_img)


            psl_padded = self._pad_to_canvas(psl_array, self.image_size)
            psl_tensor = torch.from_numpy(psl_padded.transpose(2, 0, 1)).float() / 127.5 - 1.0


            disp_path = os.path.join(self.disp_masks_dir, f'disp_{sample_id}.npy')
            disp_data = np.load(disp_path)


            sample_idx = int(sample_id)
            loads = np.load(os.path.join(self.loads_dir, f'sample_{sample_idx}.npy'))


            pf_padded = self._pad_to_canvas(pf_data, self.image_size)
            loads_padded = self._pad_to_canvas(loads, self.image_size)
            disp_padded = self._pad_to_canvas(disp_data, self.image_size)


            energy_mask = pf_padded[:, :, 1:2]


            constraints = np.concatenate([pf_padded, loads_padded], axis=2)


            constraints_tensor = torch.from_numpy(constraints.transpose(2, 0, 1)).float()
            energy_mask_tensor = torch.from_numpy(energy_mask.transpose(2, 0, 1)).float()
            disp_tensor = torch.from_numpy(disp_padded.transpose(2, 0, 1)).float()


            expected_channels = 3 + loads_padded.shape[2]
            expected_constraints_shape = (expected_channels, 64, 64)


            constraints_tensor = constraints_tensor / 127.5 - 1.0


            return {
                "target": target_tensor,
                "constraints": constraints_tensor,
                "energy_mask": energy_mask_tensor,
                "psl": psl_tensor,
                "disp": disp_tensor,
                "image_plus": image_plus_tensor,
            }

        except Exception as e:
            raise


def setup_model():


    all_args = model_and_diffusion_defaults()


    all_args.update({
        'image_size': 64,
        'num_channels': 128,
        'num_res_blocks': 3,
        'learn_sigma': True,
        'dropout': 0.3,
        'use_spatial_transformer': "true",
        'transformer_depth': 1,
        'diffusion_steps': 1000,
        'noise_schedule': "cosine",
        'use_checkpoint': True,
    })


    model, diffusion = create_model_and_diffusion(**all_args)


    if os.path.exists(config.pretrained_model_path):
        checkpoint = torch.load(config.pretrained_model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        raise FileNotFoundError(f"Pretrained model not found: {config.pretrained_model_path}")

    config.diffusion = diffusion




    target_modules = []
    for name, module in model.named_modules():

        if isinstance(module, torch.nn.Linear):

            if any(attn_key in name for attn_key in ['attn', 'attention', 'to_q', 'to_k', 'to_v', 'to_out']):
                target_modules.append(name)




    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",

        inference_mode=False,
    )


    model = get_peft_model(model, lora_config)


    for name, param in model.named_parameters():
        if 'lora_' in name or 'adapter' in name or 'peft' in name:
            if not param.requires_grad:
                param.requires_grad_(True)


    return model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    accelerator = Accelerator(
        mixed_precision="no",
        gradient_accumulation_steps=1,
    )


    model = setup_model()
    dataset = LoRATopologyDataset(config.dataset_dir, config.image_size)
    train_dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)


    optimizer = AdamW8bit(model.parameters(), lr=config.learning_rate)

    from hpgdiff.resample import create_named_schedule_sampler
    schedule_sampler = create_named_schedule_sampler("uniform", config.diffusion)


    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )


    base_dtype = None
    for name, param in model.named_parameters():
        if base_dtype is None:
            base_dtype = param.dtype
        if param.dtype != base_dtype:
            param.data = param.data.to(dtype=base_dtype)


    for name, param in model.named_parameters():
        if 'lora_' in name or 'adapter' in name or 'peft' in name:
            if param.dtype != base_dtype:
                param.data = param.data.to(dtype=base_dtype)


    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if hasattr(module, 'weight') and module.weight.dtype != base_dtype:
                module.weight.data = module.weight.data.to(dtype=base_dtype)
            if hasattr(module, 'bias') and module.bias is not None and module.bias.dtype != base_dtype:
                module.bias.data = module.bias.data.to(dtype=base_dtype)
        elif isinstance(module, torch.nn.Conv2d):
            if hasattr(module, 'weight') and module.weight.dtype != base_dtype:
                module.weight.data = module.weight.data.to(dtype=base_dtype)
            if hasattr(module, 'bias') and module.bias is not None and module.bias.dtype != base_dtype:
                module.bias.data = module.bias.data.to(dtype=base_dtype)


    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["target"]
            constraints = batch["constraints"]
            energy_mask = batch["energy_mask"]
            psl_images = batch["psl"]
            disp_masks = batch["disp"]
            image_plus = batch["image_plus"]


            clean_images = clean_images.to(device)
            constraints = constraints.to(device)
            energy_mask = energy_mask.to(device)
            psl_images = psl_images.to(device)
            disp_masks = disp_masks.to(device)
            image_plus = image_plus.to(device)


            t, _ = schedule_sampler.sample(clean_images.shape[0], device)


            micro_masks = torch.cat((energy_mask, psl_images, disp_masks), dim=1)


            try:
                losses = config.diffusion.training_losses(
                    model,
                    clean_images,
                    constraints,
                    t,
                    x_plus=image_plus,
                    model_kwargs={'context': micro_masks}
                )

                if isinstance(losses, dict):
                    loss = losses['loss']
                else:
                    loss = losses
            except Exception as e:

                try:
                    losses = config.diffusion.training_losses(
                        model,
                        clean_images,
                        constraints,
                        t,
                        x_plus=image_plus
                    )

                    if isinstance(losses, dict):
                        loss = losses['loss']
                    else:
                        loss = losses
                except Exception as e2:
                    raise e2


            if not loss.requires_grad:
                continue


            if loss.dim() > 0:
                loss = loss.mean()


            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()


            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())



    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(config.output_dir)


if __name__ == "__main__":


    main()
