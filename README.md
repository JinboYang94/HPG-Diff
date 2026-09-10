# HPG-Diff: Hierarchical Physics-Guided Diffusion with Differentiable Connectivity Constraints for Topology Optimization

## TODO

- [x] Release the training and test datasets
- [x] Update the latest code
- [ ] Add clean introduction
- [ ] Release pretrained checkpoints

Official implementation of ["HPG-Diff: Hierarchical physics-guided diffusion with differentiable connectivity constraints for topology optimization"](https://doi.org/10.1016/j.asoc.2026.115899), Jinbo Yang, Mingyue Yuan, Boyuan Zhang, Yoshifumi Kitamura, and Shikai Jing, *Applied Soft Computing*, 2026.

## 1. Introduction

HPG-Diff is a hierarchical physics-guided diffusion framework for topology optimization. It incorporates physical-field guidance and differentiable connectivity constraints to generate high-quality, structurally connected designs.

## 2. Usage

### 2.1 Prepare Data

The datasets used in the paper are available on Hugging Face:

- [HPGDiff_train](https://huggingface.co/datasets/scottyjb/HPGDiff_train)
- [HPGDiff_test](https://huggingface.co/datasets/scottyjb/HPGDiff_test)

### 2.2 Dependencies

```bash
pip install -r requirements.txt
```

### 2.3 Train and Evaluate

```bash
python scripts/diffusion_training.py
python evaluation/hpgdiff_sample.py --model_path checkpoints/model.pt --test_level 1
python evaluation/eval_checkpoints.py --dir checkpoints --model 120000 --test_level 1
cd LoRA_scripts && bash run_training.sh
```

## Citation

```bibtex
@article{yang2026hpgdiff,
  title   = {{HPG-Diff}: Hierarchical physics-guided diffusion with differentiable connectivity constraints for topology optimization},
  author  = {Yang, Jinbo and Yuan, Mingyue and Zhang, Boyuan and Kitamura, Yoshifumi and Jing, Shikai},
  journal = {Applied Soft Computing},
  volume  = {202},
  pages   = {115899},
  year    = {2026},
  doi     = {10.1016/j.asoc.2026.115899}
}
```
