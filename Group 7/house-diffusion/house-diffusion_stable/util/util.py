import argparse
import inspect

from model.respace import SpacedDiffusion, space_timesteps
from model.transformer import RPlanTransformer
from model import logger 
from model import diffusion  as gd

def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        analog_bit=False,
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="cosine",
        timestep_respacing="",
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        target_set=-1,
        set_name='',
    )

def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
            dataset='',
            use_checkpoint=False,
            input_channels=0,
            condition_channels=0,
            out_channels=0,
            use_unet=False,
            num_channels=128
            )
    res.update(diffusion_defaults())
    return res

def create_model_and_diffusion(
    input_channels,
    condition_channels,
    num_channels,
    out_channels,
    learn_sigma,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    analog_bit = False
):
    model = RPlanTransformer(input_channels, condition_channels, num_channels, out_channels, "rplan", analog_bit)

    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion

def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    
    if rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )




def check_overflow(value):
    return (value == float("inf")) or (value == -float("inf")) or (value != value)


