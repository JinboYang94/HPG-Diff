import argparse
import torch as th
import time

import hpgdiff.dist_util as dist_util
import hpgdiff.logger as logger
from hpgdiff.image_datasets_diffusion_model import load_data
from hpgdiff.resample import create_named_schedule_sampler
from hpgdiff.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from hpgdiff.train_util import TrainLoop


def main():
    start_time = time.time()
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    all_args = model_and_diffusion_defaults()
    all_args.update(args_to_dict(args, model_and_diffusion_defaults().keys()))
    all_args.update(vars(args))

    logger.log("Training Hyperparameters:")
    for key, value in sorted(all_args.items()):
        logger.log(f"{key}: {value}")
    logger.log("-" * 30)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"Total parameters: {total_params:,}")
    logger.log(f"Trainable parameters: {trainable_params:,}")
    logger.log(f"Frozen parameters: {total_params - trainable_params:,}")

    logger.log(f"Training device: {dist_util.dev()}")
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        batch_size=args.batch_size,
        image_size=args.image_size,
    )

    logger.log("training...")
    try:
        TrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=args.batch_size,
            microbatch=args.microbatch,
            lr=args.lr,
            ema_rate=args.ema_rate,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            resume_checkpoint=args.resume_checkpoint,
            use_fp16=args.use_fp16,
            fp16_scale_growth=args.fp16_scale_growth,
            schedule_sampler=schedule_sampler,
            weight_decay=args.weight_decay,
            lr_anneal_steps=args.lr_anneal_steps,
        ).run_loop()
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"\nTraining failed: {str(e)}")
        raise e
    finally:
        end_time = time.time()
        total_time = end_time - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = total_time % 60

        logger.info(f"\nTotal runtime: {hours}h {minutes}m {seconds:.2f}s")


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        use_spatial_transformer=False,
        transformer_depth=1,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
