import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
from ddpm import DDPMSampler
from pipeline import get_time_embedding
from dataloader import train_dataloader
import model_loader
import time
from config import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_file = "./data/v1-5-pruned.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

vae = models['encoder']
text_encoder = models['clip']
decoder = models['decoder']
unet = models['diffusion']
ddpm = DDPMSampler(generator=None)

# Disable gradient computations for the VAE, DDPM, and text_encoder models
for param in vae.parameters():
    param.requires_grad = False

for param in text_encoder.parameters():
    param.requires_grad = False

# set the vae and text_encoder to eval mode
vae.eval()
text_encoder.eval()

optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate, betas=(adam_beta1, adam_beta2), weight_decay=adam_weight_decay, eps=adam_epsilon)


def train(num_train_epochs, device="cuda", save_steps=1000, max_train_steps=10000):
    global_step = 0

    # create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # move models to the device
    vae.to(device)
    text_encoder.to(device)
    unet.to(device)

    num_train_epochs = tqdm(range(first_epoch, num_train_epochs), desc="Epoch")
    for epoch in num_train_epochs:
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            start_time = time.time()

            # batch consists of images and texts, we need to extract the images and texts

            # move batch to the device
            batch["pixel_values"] = batch["pixel_values"].to(device)
            batch["input_ids"] = batch["input_ids"].to(device)

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            encoder_noise = torch.randn(latents_shape, device=device)
            encoder_noise = encoder_noise.to(device)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = vae(batch["pixel_values"], encoder_noise)

            # Sample noise that we'll add to the latents -> it is done inside the add noise method
            # noise = torch.randn_like(latents)
            
            bsz = latents.shape[0]

            # Sample a random timestep for each image and text
            timesteps = torch.randint(0, ddpm.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            text_timesteps = torch.randint(0, ddpm.num_train_timesteps, (bsz,), device=latents.device)
            text_timesteps = text_timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep (this is the forward diffusion process)
            noisy_latents, image_noise = ddpm.add_noise(latents, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(batch["input_ids"])

            # Add noise to the text query according to the noise magnitude at each timestep
            noisy_text_query, text_noise = ddpm.add_noise(encoder_hidden_states, text_timesteps)

            image_time_embeddings = get_time_embedding(timesteps, is_image=True).to(device)
            text_time_embeddings = get_time_embedding(timesteps, is_image=False).to(device)
            
            # take average and normalize the text time embeddings
            average_noisy_text_query = noisy_text_query.mean(dim=1)
            text_query = F.normalize(average_noisy_text_query, p=2, dim=-1)

            # Target for the model is the noise that was added to the latents and the text query
            image_target = image_noise
            text_target = text_query

            # Predict the noise residual and compute loss
            image_pred, text_pred = unet(noisy_latents, encoder_hidden_states, image_time_embeddings, text_time_embeddings, text_query)

            image_loss = F.mse_loss(image_pred.float(), image_target.float(), reduction="mean")
            text_loss = F.mse_loss(text_pred.float(), text_target.float(), reduction="mean")
            
            train_loss += image_loss + Lambda * text_loss

            # Backpropagate
            loss = image_loss + Lambda * text_loss
            loss.backward()

            optimizer.zero_grad()
            optimizer.step()
            # lr_scheduler.step() # maybe linear scheduler can be added

            if global_step % save_steps == 0:
                # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                if checkpoints_total_limit is not None:
                    checkpoints = os.listdir(output_dir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                    if len(checkpoints) >= checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]

                        print(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        )
                        print(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                            os.remove(removing_checkpoint)

                save_path = os.path.join(output_dir, f"checkpoint-{global_step}")

                # Save model state and optimizer state
                torch.save({
                    'model_state_dict': unet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, save_path)

                print(f"Saved state to {save_path}")

            end_time = time.time()
            print("step_loss:", loss.detach().item(), "time per step:", (end_time - start_time) / bsz, "step per second:", bsz / (end_time - start_time))
            
            if global_step >= max_train_steps:
                break

            global_step += 1

        print("Average loss over epoch:", train_loss / (step + 1))


if __name__ == "__main__":
    train(num_train_epochs)
