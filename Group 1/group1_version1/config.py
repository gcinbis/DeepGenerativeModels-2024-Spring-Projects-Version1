# define the constants 
WIDTH = 256
HEIGHT = 256
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8
BATCH_SIZE = 16

# training parameters
first_epoch = 0
num_train_epochs = 10
latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)
Lambda = 1.0

# optimizer parameters
learning_rate = 1e-4
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_weight_decay = 0.0
adam_epsilon = 1e-8

# checkpoint parameters
checkpoints_total_limit = 1
output_dir = "output"