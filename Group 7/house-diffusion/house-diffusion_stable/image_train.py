"""
Train a diffusion model on images.
"""

import argparse

from model import dist_util, logger
from dataset import load_rplanhg_data
from model.sample import create_named_schedule_sampler
from util.util import (
    create_model_and_diffusion,
)
from model.util import TrainLoop


def main():
    num_channels = 1024
    num_coords = 2
    input_channels = 18
    condition_channels = 89
    out_channels = num_coords * 1

    logger.configure()
    logger.log("creating models...")

    model, diffusion = create_model_and_diffusion(
        input_channels = input_channels,
        condition_channels = condition_channels,
        num_channels = num_channels,
        out_channels = out_channels,
        learn_sigma = False,
        diffusion_steps=1000,
        noise_schedule="cosine",
        timestep_respacing="",
        predict_xstart = False,
        rescale_timesteps = False,
        rescale_learned_sigmas = False,
        analog_bit = False
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler("uniform", diffusion)

    logger.log("creating data loader...")

    data = load_rplanhg_data(
        batch_size=128,
        analog_bit=False,
        target_set=8,
        set_name="train",
        num_points=10
    )
    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=128,
        microbatch=1,
        lr=1e-4,
        ema_rate=0.9999,
        log_interval=10,
        save_interval=10,
        resume_checkpoint="",
        schedule_sampler=schedule_sampler,
        weight_decay=0,
        lr_anneal_steps=100,
        analog_bit=False,
    ).run_loop()


if __name__ == "__main__":
    main()
