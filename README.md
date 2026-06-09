# HPG-Diff

## Environment
To setup the the environment to run the code, create a new Python environment and run:
```bash
pip install -r requirements.txt
```

## Usage

Training:

```bash
python scripts/diffusion_training.py
```

Inference:

```bash
python evaluation/hpgdiff_sample.py --model_path checkpoints/model.pt --test_level 1
```

Evaluation:

```bash
python evaluation/eval_checkpoints.py --dir ../checkpoints/ --model "model" --test_level "1, 2"
```



LoRA fine-tuning:

```bash
bash LoRA_scripts/run_training.sh
```

LoRA evaluation:

```bash
python evaluation/eval_lora_checkpoints.py --base_model_path ../checkpoints/model.pt --lora_path ../LoRA_scripts/lora_checkpoints --test_level "1, 2"
```

## Training Hyperparameters
The optimization hyperparameters are selected for stable full-model training and memory feasibility:

HPG-Diff: image size `64`, channel size `128`, `3` residual blocks, learned sigma, dropout `0.3`, cosine noise schedule, `1000` diffusion steps, FP16 enabled, spatial transformer enabled, transformer depth `1`, batch size `64`, and  learning rate `1e-4`.

LoRA:  batch size `40`, `20` epochs, learning rate `2e-5`, rank `8`, alpha `16`, and ``dropout `0.1`.

## Inference Details

Inference uses channel size `128`, `3` residual blocks, learned sigma, dropout `0.3`, FP16, cosine noise schedule, `1000` training diffusion steps, and timestep respacing `100` for sampling. 

Generated samples are scaled from model output to `uint8` by multiplying by `255`, clamping to `[0, 255]`, and saving as `samples_{N}x64x64x1.npz`.

Evaluation binarizes generated samples by setting values `< 127` to `0` and values `>= 128` to `255`.

The FMS propagation uses hard `3 x 3` max pooling for `K` iterations. With `N = H x W` design elements, the complexity is `O(KN)`. For `64 x 64` domains, we use `K = 64`; this is a simple local pooling operation and is small compared with the diffusion UNet forward/backward cost.

LoRA target layers are limited to attention-related linear modules whose names contain one of `attn`, `attention`, `to_q`, `to_k`, `to_v`, or `to_out`.
