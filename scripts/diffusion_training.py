import os
import sys
import argparse
import time
from pathlib import Path

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='./checkpoints')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--save_interval', type=int, default=10000)
    parser.add_argument('--use_spatial_transformer', type=str, default="true")
    parser.add_argument('--transformer_depth', type=int, default=1)

    return parser

def get_next_run_number(base_dir):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    existing_runs = [d for d in os.listdir(base_dir)
                    if os.path.isdir(os.path.join(base_dir, d))
                    and d.startswith('diff_logdir_')]

    if not existing_runs:
        return 1

    run_numbers = [int(d.split('_')[-1]) for d in existing_runs]
    return max(run_numbers) + 1

def main():
    args, _ = create_argparser().parse_known_args()
    start_time = time.time()
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    base_checkpoint_dir = Path(args.dir)
    if not base_checkpoint_dir.is_absolute():
        base_checkpoint_dir = project_root / base_checkpoint_dir
    lr = args.lr
    weight_decay = args.weight_decay
    dropout = args.dropout
    save_interval = args.save_interval
    use_spatial_transformer = args.use_spatial_transformer.lower()
    transformer_depth = args.transformer_depth

    os.makedirs(base_checkpoint_dir, exist_ok=True)
    next_run = get_next_run_number(str(base_checkpoint_dir))
    log_dir = base_checkpoint_dir / f'diff_logdir_{next_run}'
    os.environ['HPGDIFF_LOGDIR'] = str(log_dir)
    os.makedirs(log_dir, exist_ok=True)


    train_flags = f"--batch_size 64 --save_interval {save_interval} --use_fp16 True --lr {lr} --weight_decay {weight_decay}"
    model_flags = f"--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout {dropout}"
    diffusion_flags = "--diffusion_steps 1000 --noise_schedule cosine"
    model_flags += f" --use_spatial_transformer {use_spatial_transformer}"
    if transformer_depth != 1:
        model_flags += f" --transformer_depth {transformer_depth}"

    try:
        os.environ['HPGDIFF_DATASET'] = 'huggingface'
        train_script = project_root / "scripts" / "image_train.py"
        if not os.path.exists(train_script):
            raise FileNotFoundError(f"Training script {train_script} does not exist.")

        command = f"{train_script} {model_flags} {diffusion_flags} {train_flags}"
        from scripts.image_train import main as train_main
        sys.argv = [str(train_script)] + f"{model_flags} {diffusion_flags} {train_flags}".split()
        train_main()

    except KeyboardInterrupt:
        return
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        end_time = time.time()
        total_time = end_time - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)


if __name__ == "__main__":
    main()
