import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

from model import HouseDiffTransformer
from utils import broadcast_into, dec2bin

class DiffusionModel(pl.LightningModule):
    class DiffusionModel(nn.Module):
        def __init__(
                self,
                num_timesteps=1000,
                beta_schedule="cosine",
                model_mean_type="epsilon",
                model_var_type="fixed_large",
                model_kwargs=None
        ):
            """
            Initializes a DiffusionModel instance.

            Args:
                num_timesteps (int): The number of diffusion timesteps.
                beta_schedule (str): The schedule for beta values.
                model_mean_type (str): The type of model mean.
                model_var_type (str): The type of model variance.
                model_kwargs (dict): Additional keyword arguments for the HouseDiffTransformer model.

            """
            super(DiffusionModel, self).__init__()

            if model_kwargs is None:
                model_kwargs = dict(
                    in_channels=18,
                    condition_channels=89,
                    model_channels=1024,
                    out_channels=2,
                )

            self.num_timesteps = num_timesteps
            self.beta_schedule = beta_schedule
            self.model_mean_type = model_mean_type
            self.model_var_type = model_var_type

            self.model = HouseDiffTransformer(**model_kwargs)

            self.save_hyperparameters()

            betas = self.get_beta_schedule()
            self.register_buffer("betas", betas)
            alphas_cumprod = torch.cumprod(1 - betas, dim=0)
            self.register_buffer("alphas_cumprod", alphas_cumprod)
            alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
            self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

            self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
            self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
            self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
            self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(alphas_cumprod))
            self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))

            # Put posterior means and variances too
            posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
            self.register_buffer("posterior_var", posterior_var)
            self.register_buffer("posterior_log_variance_clipped", torch.log(torch.cat([posterior_var[1:2], posterior_var[1:]])))

            posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
            self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
            posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(1.0 - betas) / (1.0 - alphas_cumprod)
            self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    def get_beta_schedule(self):
        """
        Returns the beta schedule based on the specified beta_schedule.

        If beta_schedule is "linear", the beta values are linearly spaced between
        scale * 1e-4 and scale * 2e-2, where scale is calculated as 1000 divided by
        the number of timesteps.

        If beta_schedule is "cosine", the beta values are calculated using a cosine
        function. The alpha_bar values are computed based on a linspace from 0 to 1,
        and then the betas are calculated as the minimum of 1 - alpha_bar[1:] / alpha_bar[:-1]
        and 0.999.

        Returns:
            torch.Tensor: The beta schedule.
        """
        if self.beta_schedule == "linear":
            scale = 1000 / self.num_timesteps
            return torch.linspace(scale * 1e-4, scale * 2e-2, self.num_timesteps)
        if self.beta_schedule == "cosine":
            t = torch.linspace(0, 1, self.num_timesteps + 1)
            alpha_bar = torch.cos(t * torch.pi / 2) ** 2
            betas = torch.min(1 - alpha_bar[1:] / alpha_bar[:-1], torch.tensor(0.999))
            return betas
        
    def q_sample(self, x_start, t, noise):
        """
        Sample from the q distribution (forward process) at a specific timestep t.
        
        Args:
            x_start (torch.Tensor): The original data (batch of images).
            t (torch.Tensor): The timestep(s) at which to add noise (can be a single value or a tensor of timesteps).
        
        Returns:
            torch.Tensor: The noisy image at timestep t.
        """

        # The noisy sample is a combination of the original data and added noise
        x_noisy = broadcast_into(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start 
        + broadcast_into(self.sqrt_one_minus_alphas_cumprod, t, noise.shape) * noise

        return x_noisy
    
    def forward(self, x, t, x_t_alpha, eps_alpha, conditions):
        """
        Perform a forward pass through the model.
        
        Args:
            x (torch.Tensor): The input data (batch of images).
            t (torch.Tensor): The timestep(s) at which to add noise (can be a single value or a tensor of timesteps).
            x_t_alpha (torch.Tensor): The product of the square root of the cumulative alphas and the input data.
            eps_alpha (torch.Tensor): The product of the square root of the reciprocal of the cumulative alphas minus 1 and the noise.
        
        Returns:
            torch.Tensor: The output of the model.
        """
        out_decimal, out_binary = self.model(x, t, x_t_alpha, eps_alpha, **conditions)
        return out_decimal, out_binary
    
    def compute_loss(self, x_start, t, conditions, noise=None):
        """
        Calculate the loss for a given batch of data at a specific timestep t.
        
        Args:
            x_start (torch.Tensor): The original data (batch of images).
            t (torch.Tensor): The timestep(s) at which to add noise (can be a single value or a tensor of timesteps).
            noise (torch.Tensor): The noise to add to the data.
        
        Returns:
            torch.Tensor: The loss for the batch at timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_start).type_as(x_start)

        x_noisy = self.q_sample(x_start, t, noise)
        
        x_t_alpha = broadcast_into(self.sqrt_recip_alphas_cumprod, t, x_start.shape).permute([0, 2, 1])
        eps_alpha = broadcast_into(self.sqrt_recipm1_alphas_cumprod, t, x_start.shape).permute([0, 2, 1])
        out_decimal, out_binary = self.forward(x_noisy, t, x_t_alpha, eps_alpha, conditions)

        target_decimal = noise

        target_binary = x_start.detach()
        target_binary = ((target_binary / 2 + 0.5) * 256).permute([0, 2, 1]).round().int()
        target_binary = dec2bin(target_binary, 8).reshape([target_decimal.shape[0], target_decimal.shape[2], 16]).permute([0,2,1])
        target_binary[target_binary == 0] = -1

        tmp_mask = (1 - conditions['src_key_padding_mask']).unsqueeze(1)

        mse_binary = F.mse_loss(out_binary * tmp_mask, target_binary * tmp_mask) # / tmp_mask.sum() # Is this correct?
        mse_decimal = F.mse_loss(out_decimal * tmp_mask, target_decimal * tmp_mask) # / tmp_mask.sum() # Is this correct?

        return {
            "mse_binary": mse_binary,
            "mse_decimal": mse_decimal, 
            "loss": mse_binary + mse_decimal
        }


    def training_step(self, batch, batch_idx):
        """
        Perform a training step with the given batch.

        Args:
            batch (list): A batch from the dataloader, typically containing images and possibly other data.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The loss for the batch.
        """
        # Get batch
        x_start, conditions = batch
        # Sample timestep t for the batch
        t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=self.device).long()
        
        losses = self.compute_loss(x_start, t, conditions)

        self.log("train_loss", losses["loss"])
        self.log("train_mse_decimal", losses["mse_decimal"])
        self.log("train_mse_binary", losses["mse_binary"])

        return losses["loss"]
    
    def configure_optimizers(self):
        """
        Configure the optimizer for training.

        Returns:
            torch.optim.Optimizer: The optimizer to use for training.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer