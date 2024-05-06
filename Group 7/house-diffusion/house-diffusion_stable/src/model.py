import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils import dec2bin

class TimestepEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        """
        Initialize the TimestepEmbedding module.

        Args:
            dim (int): The dimension of the embedding.
            max_period (int, optional): The maximum period for the sinusoidal functions. Defaults to 10000.
        """
        super(TimestepEmbedding, self).__init__()
        # Precompute frequency terms and register as buffer
        half_dim = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half_dim, dtype=torch.float32) / half_dim)
        self.register_buffer('freqs', freqs)
        self.register_buffer('padding', dim % 2)

        self.time_embeddings = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, timesteps):
        """
        Forward pass of the TimestepEmbedding module.

        Args:
            timesteps (torch.Tensor): The input timesteps. Shape: (batch_size, sequence_length).

        Returns:
            torch.Tensor: The embedded timesteps. Shape: (batch_size, sequence_length, embedding_dim).
        """
        # Calculate arguments for sinusoidal functions
        args = timesteps[:, None] * self.freqs[None]

        # Compute sinusoidal embeddings
        embeddings = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        # Pad embeddings to the correct size
        embeddings = F.pad(embeddings, (0, self.padding), "constant", 0)

        return self.time_embeddings(embeddings)
    
class HouseDiffEncoder(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        """
        Initialize the HouseDiffEncoder module.

        Args:
            dim (int): The dimension of the input and output tensors.
            num_heads (int): The number of attention heads.
            dropout (float, optional): The dropout probability. Defaults to 0.1.
        """
        super(HouseDiffEncoder, self).__init__()
        mha_kwargs = dict(dim=dim, num_heads=num_heads, dropout=dropout)
        
        self.self_attention = nn.MultiheadAttention(**mha_kwargs)
        self.door_attention = nn.MultiheadAttention(**mha_kwargs)
        self.gen_attention = nn.MultiheadAttention(**mha_kwargs)

        self.norm_1 = nn.InstanceNorm1d(dim) # TODO: nn.LayerNorm(dim) # can be used too.
        self.norm_2 = nn.InstanceNorm1d(dim) # TODO: nn.LayerNorm(dim) # can be used too.

        self.dropout = nn.Dropout(dropout)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, dim),
        )

        self.attention = lambda attention, x, mask: self.dropout(attention(x, x, x, attn_mask=mask))

    def forward(self, x, self_mask, door_mask, gen_mask):
        """
        Forward pass of the HouseDiffEncoder module.

        Args:
            x (torch.Tensor): The input tensor. Shape: (batch_size, sequence_length, dim).
            self_mask (torch.Tensor): The self-attention mask. Shape: (sequence_length, sequence_length).
            door_mask (torch.Tensor): The door-attention mask. Shape: (sequence_length, sequence_length).
            gen_mask (torch.Tensor): The gen-attention mask. Shape: (sequence_length, sequence_length).

        Returns:
            torch.Tensor: The output tensor. Shape: (batch_size, sequence_length, dim).
        """
        # TODO: Assertion for gen_mask needs to be added.
        norm_x = self.norm_1(x)

        self_attended = self.attention(self.self_attention, norm_x, self_mask)
        door_attended = self.attention(self.door_attention, norm_x, door_mask)
        gen_attended = self.attention(self.gen_attention, norm_x, gen_mask)

        attended = x + self_attended + door_attended + gen_attended
        norm_attended = self.norm_2(attended)

        return norm_attended + self.dropout(self.mlp(norm_attended))
    
class HouseDiffTransformer(nn.Module):
    def __init__(
            self, 
            in_channels: int,
            condition_channels: int,
            model_channels: int,
            out_channels: int,
            condition_keys: list[str] = ['room_types, corner_indices, room_indices'], 
        ) -> None:
        """
        Initialize the HouseDiffTransformer module.

        Args:
            in_channels (int): The number of input channels.
            condition_channels (int): The number of condition channels.
            model_channels (int): The number of channels in the model.
            out_channels (int): The number of output channels.
            condition_keys (list[str], optional): The keys for the condition embeddings. Defaults to ['room_types, corner_indices, room_indices'].
        """
        super(HouseDiffTransformer, self).__init__()

        self.condition_keys = condition_keys

        self.timestep_embeddings = TimestepEmbedding(model_channels)

        self.input_embeddings = nn.Linear(in_channels, model_channels)
        self.condition_embeddings = nn.Linear(condition_channels, model_channels)
        
        self.transformer_layers = nn.ModuleList([
            HouseDiffEncoder(model_channels, num_heads=4)
            for _ in range(4) # TODO: Can make this a hyperparameter.
        ])

        self.analog_mlp = nn.Sequential(
            nn.Linear(model_channels, model_channels),
            nn.ReLU(),
            nn.Linear(model_channels, model_channels // 2),
            nn.Linear(model_channels // 2, out_channels),
        )

        self.binary_mlp = nn.Sequential(
            nn.Linear(162 + model_channels, model_channels), # TODO: WTF is 162?
            nn.ReLU(),
            HouseDiffEncoder(model_channels, num_heads=1),
            HouseDiffEncoder(model_channels, num_heads=1),
            nn.Linear(model_channels, 16), # TODO: WTF is 16?
        )

    def expand_points(self, points, connections):
        """
        Expands the given set of points by interpolating new points between each pair of connected points.

        Args:
            points (torch.Tensor): The original set of points. Shape: (batch_size, num_points, 2).
            connections (torch.Tensor): The indices of connected points. Shape: (batch_size, num_connections, 2).

        Returns:
            torch.Tensor: The expanded set of points. Shape: (batch_size, num_points, 2).
        """
        average_points = lambda a, b: (a + b) / 2

        # Get the original shape of points
        original_shape = points.shape

        # Reshape points to separate x and y for easier manipulation
        points = points.view([points.shape[0], points.shape[1], 2, -1])

        # Retrieve connected points using indices from connections
        connected_points = points[torch.arange(points.shape[0])[:, None], connections[:,:,1].long()]
        connected_points = connected_points.view([connected_points.shape[0], connected_points.shape[1], 2, -1])

        # Interpolate new points
        mid_points = average_points(points, connected_points)
        quarter_points = average_points(points, mid_points)
        three_quarter_points = average_points(mid_points, connected_points)

        # Concatenate all points to form the new expanded set
        expanded_points = torch.cat([
            points.view(original_shape),
            average_points(quarter_points, points).view(original_shape), # Halfway between points and quarter_points
            quarter_points.view(original_shape),
            average_points(mid_points, quarter_points).view(original_shape), # Halfway between quarter_points and mid_points
            mid_points.view(original_shape),
            average_points(three_quarter_points, mid_points).view(original_shape), # Halfway between mid_points and three_quarter_points
            three_quarter_points.view(original_shape),
            average_points(connected_points, three_quarter_points).view(original_shape), # Halfway between three_quarter_points and connected_points
            connected_points.view(original_shape)
        ], 2)

        return expanded_points.detach()
    
    def compute_binary_output(self, x, output_decimal, xtalpha, epsalpha, condition_embeddings):
        """
        Compute the binary output of the model.

        Args:
            x (torch.Tensor): The input tensor. Shape: (batch_size, sequence_length, dim).
            output_decimal (torch.Tensor): The decimal output tensor. Shape: (batch_size, sequence_length, out_channels).
            xtalpha (torch.Tensor): TODO: Add description. Shape: (batch_size, sequence_length, 9).
            epsalpha (torch.Tensor): TODO: Add description. Shape: (batch_size, sequence_length, 9).
            condition_embeddings (torch.Tensor): The condition embeddings. Shape: (batch_size, sequence_length, model_channels).

        Returns:
            torch.Tensor: The binary output tensor. Shape: (batch_size, sequence_length, 16).
        """
        # TODO: WTF?!
        output_binary_start = x * xtalpha.repeat([1, 1, 9]) - output_decimal.repeat([1, 1, 9]) * epsalpha.repeat([1, 1, 9])
        output_binary = dec2bin((output_binary_start / 2 + 0.5) * 256)

        output_binary = torch.cat([
            output_binary_start,
            output_binary.reshape([x.shape[0], x.shape[1], 16*9]),
            condition_embeddings
        ], 2)

        return self.binary_mlp(output_binary)
        
    def forward(self, x, timesteps, xtalpha, epsalpha, **kwargs):
        """
        Forward pass of the HouseDiffTransformer module.

        Args:
            x (torch.Tensor): The input tensor. Shape: (batch_size, sequence_length, in_channels).
            timesteps (torch.Tensor): The input timesteps. Shape: (batch_size, sequence_length).
            xtalpha (torch.Tensor): TODO: Add description. Shape: (batch_size, sequence_length, 9).
            epsalpha (torch.Tensor): TODO: Add description. Shape: (batch_size, sequence_length, 9).
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The decimal output tensor. Shape: (batch_size, out_channels, sequence_length).
            torch.Tensor: The binary output tensor. Shape: (batch_size, 16, sequence_length).
        """
        # TODO: rename xtalpha, epsalpha
        # TODO: remove is_syn related nonsenses (by editing the data loader)
        # TODO: kwargs is being used pretty badly here. Rethink this. (by editing the data loader)

        x = x.permute(0, 2, 1).float()

        x = self.expand_points(x, kwargs['connections'])

        input_embeddings = self.input_embeddings(x)
        time_embeddings = self.timestep_embeddings(timesteps).unsqueeze(1).repeat([1, x.shape[1], 1]) # or input_embeddings.shape[1]

        condition_embeddings = self.condition_embeddings(
            torch.cat([kwargs[key] for key in self.condition_keys], 2).float()
        )

        x = input_embeddings + time_embeddings + condition_embeddings

        for layer in self.transformer_layers:
            x = layer(x, kwargs['door_mask'], kwargs['self_mask'], kwargs['gen_mask'])

        output_decimal = self.analog_mlp(x)
        output_binary = self.compute_binary_output(x, output_decimal, xtalpha, epsalpha, condition_embeddings)

        return output_decimal.permute([0, 2, 1]), output_binary.permute([0, 2, 1]) # convert back to [N x S x C] to [N x C x S]
