import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset

evaluation_root = Path(__file__).resolve().parent
project_root = evaluation_root.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(evaluation_root) not in sys.path:
    sys.path.insert(0, str(evaluation_root))

from hpgdiff import logger
import hpgdiff_analysis

os.environ['HPGDIFF_LOGDIR'] = str(project_root / 'generated_rmt')
os.makedirs(os.environ['HPGDIFF_LOGDIR'], exist_ok=True)


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='../checkpoints/local_loss_0.01')
    parser.add_argument('--model', type=str, default='120000')
    parser.add_argument('--test_level', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--use_spatial_transformer', type=str, default='true')
    parser.add_argument('--transformer_depth', type=int, default=1)
    parser.add_argument('--psl_attn_sample_index', type=int, default=-1)
    parser.add_argument('--psl_attn_dir', type=str, default='feature_maps_psl_attn')
    parser.add_argument('--psl_attn_max', type=int, default=64)
    parser.add_argument('--psl_attn_t_max', type=int, default=50)
    parser.add_argument('--attn_branch', type=str, default='psl', choices=['psl', 'u', 'energy'])
    return parser


def get_next_log_dir():
    base_dir = os.environ['HPGDIFF_LOGDIR']
    existing_dirs = glob.glob(os.path.join(base_dir, 'eval_log_*'))
    if not existing_dirs:
        return os.path.join(base_dir, 'eval_log_1')
    existing_numbers = [int(d.split('_')[-1]) for d in existing_dirs]
    next_number = max(existing_numbers) + 1
    return os.path.join(base_dir, f'eval_log_{next_number}')


def load_test_constraints(test_level):
    try:
        dataset = load_dataset('test')
        test_split = f'test_level_{test_level}'
        test_data = dataset[test_split]
        size = 1800 if test_level == 1 else 1000
        constraints = np.empty(size, dtype=object)
        for i in range(size):
            sample = test_data[i]
            summary = sample['summary']
            bc_conf = []
            bc_data = summary['BC_conf']
            for j, node_list in enumerate(bc_data['0']):
                nodes = list(map(int, node_list))
                bc_type = int(bc_data['1'][j])
                bc_conf.append((nodes, bc_type))
            constraints[i] = {
                'y_loads': np.array(summary['y_loads'], dtype=np.float64),
                'BC_conf': bc_conf,
                'VF': float(summary['VF']),
                'load_nodes': np.array(summary['load_nodes'], dtype=np.float64),
                'BC_conf_x': summary['BC_conf_x'],
                'BC_conf_y': summary['BC_conf_y'],
                'load_coord': np.array(summary['load_coord'], dtype=np.float64),
                'x_loads': np.array(summary['x_loads'], dtype=np.float64),
            }
        logger.info(f"Constraints loaded. Dataset size: {len(constraints)}")
        return constraints
    except Exception as e:
        logger.error(f"Failed to load test constraints: {str(e)}")
        raise


def generate_test_samples(args, num_samples):
    model_flags = '--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3 --use_fp16 True'
    diffusion_flags = '--diffusion_steps 1000 --timestep_respacing 100 --noise_schedule cosine'
    data_flags = f'--test_level {args.test_level} --num_samples {num_samples} --batch_size {args.batch_size}'
    transformer_flags = f'--use_spatial_transformer {args.use_spatial_transformer} --transformer_depth {args.transformer_depth}'
    psl_attn_flags = (
        f'--psl_attn_sample_index {args.psl_attn_sample_index} '
        f'--psl_attn_dir {args.psl_attn_dir} '
        f'--psl_attn_max {args.psl_attn_max} '
        f'--psl_attn_t_max {args.psl_attn_t_max} '
        f'--attn_branch {args.attn_branch}'
    )
    checkpoint_dir = Path(args.dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = evaluation_root / checkpoint_dir
    model_path = checkpoint_dir / f'model{args.model}.pt'
    checkpoints_flags = f'--model_path {model_path}'
    try:
        os.environ['HPGDIFF_LOGDIR'] = str(project_root / 'generated_rmt')
        sample_script = evaluation_root / 'hpgdiff_sample.py'
        if not os.path.exists(sample_script):
            raise FileNotFoundError(f"Sample script {sample_script} does not exist.")
        command_args = f'{model_flags} {diffusion_flags} {checkpoints_flags} {data_flags} {transformer_flags} {psl_attn_flags}'
        logger.info('Generating samples...')
        logger.info(f"Command:\n{sample_script} {command_args}\n")
        from hpgdiff_sample import main as generate_main
        sys.argv = [str(sample_script)] + command_args.split()
        generate_main()
        logger.info('\nGeneration complete.')
        return True
    except KeyboardInterrupt:
        logger.info('\nGeneration interrupted by user.')
        return False
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def evaluate_samples(constraints, gen_dir, num_samples, num_folder, test_level):
    analysis = hpgdiff_analysis.hpgdiff_analysis(num_samples, num_folder, constraints, gen_dir)
    dataset = load_dataset('test')
    test_data = dataset[f'test_level_{test_level}']
    compliance_opt = np.array(test_data[0]['compliance'])
    logger.info(f"Reference compliance loaded. Shape: {compliance_opt.shape}")
    compliance_opt = hpgdiff_analysis.re_order_tab(num_samples, compliance_opt)
    analysis_relative = (
        analysis[0] / compliance_opt - 1,
        analysis[1],
        analysis[2],
        analysis[3],
    )
    logger.info('\nEvaluation results:')
    hpgdiff_analysis.print_results(analysis_relative)
    return analysis, compliance_opt, analysis_relative


def evaluate_one_checkpoint(args):
    start_time = time.time()
    log_dir = get_next_log_dir()
    os.makedirs(log_dir, exist_ok=True)
    logger.configure(log_dir)
    folder_num_pool = [1800, 1000]
    gen_dir = log_dir + '/'
    num_samples = 1800 if args.test_level == 1 else 1000
    num_folder = folder_num_pool[args.test_level - 1]
    try:
        logger.info('Loading test constraints...')
        constraints = load_test_constraints(args.test_level)
        logger.info(f"Processing model: {args.model}, test level: {args.test_level}")
        sample_start_time = time.time()
        res = generate_test_samples(args, num_samples)
        sample_duration = time.time() - sample_start_time
        per_sample_time = sample_duration / num_samples
        logger.info(f"Generation stats: total={sample_duration:.2f}s, samples={num_samples}, per_sample={per_sample_time:.3f}s")
        if not res:
            logger.error(f"Sample generation failed (model: {args.model}, test level: {args.test_level})")
            return False
        logger.info('\nEvaluating generated samples...')
        evaluate_samples(constraints, gen_dir, num_samples, num_folder, test_level=args.test_level)
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = total_time % 60
        logger.info(f"\nTotal runtime: {hours}h {minutes}m {seconds:.2f}s")
        return True
    except Exception as e:
        logger.error(f"Failed to process checkpoint: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    logger.configure()
    args, _ = create_argparser().parse_known_args()
    success = evaluate_one_checkpoint(args)
    logger.info(f"Status: {'success' if success else 'failed'}")


if __name__ == '__main__':
    main()
