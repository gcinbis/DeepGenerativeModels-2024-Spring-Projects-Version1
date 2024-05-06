import sys
sys.path.append("/workspace/src")

from dataset import RPlanDataModule
from diffusion import DiffusionModel
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

def train():
    # Initialize the data module
    data_module = RPlanDataModule(
        batch_size=32,
        root_folder="rplan_json"
    )

    # Initialize the diffusion model
    diffusion_model = DiffusionModel(
        model_kwargs=dict(
            in_channels=18,
            condition_channels=89,
            model_channels=1024,
            out_channels=2
        )
    )

    # Initialize the TensorBoard logger
    logger = TensorBoardLogger("tb_logs", name="house_diffusion")

    # Initialize the Lightning trainer
    trainer = pl.Trainer(devices=1, logger=logger)

    # Train the model
    trainer.fit(diffusion_model, data_module)

if __name__ == "__main__":
    train()