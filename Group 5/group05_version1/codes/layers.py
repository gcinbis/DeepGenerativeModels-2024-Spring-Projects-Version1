import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from einops import rearrange



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        """
        Initializes the Mlp class.

        Parameters:
        in_features (int): Number of input features.
        hidden_features (int, optional): Number of hidden layer features. Defaults to in_features if not provided.
        out_features (int, optional): Number of output features. Defaults to in_features if not provided.
        act_layer (torch.nn.Module, optional): Activation function to use. Defaults to ReLU.
        drop (float, optional): Dropout probability. Defaults to 0.0.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # First fully connected layer
        self.fc1 = nn.Linear(in_features, hidden_features)

        # Activation layer
        self.act = act_layer()

        # Second fully connected layer
        self.fc2 = nn.Linear(hidden_features, out_features)

        # Dropout layer
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Forward pass of the network.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after passing through the MLP.
        """
        # First fully connected layer followed by activation and dropout
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        # Second fully connected layer followed by dropout
        x = self.fc2(x)
        x = self.drop(x)
        
        return x


class DecoderSelfMSA(nn.Module):
    def __init__(self, emb_size, num_heads, window_size=8, shifted=True):
        """
        Implements a shifted window self-attention mechanism for content features.

        Parameters:
        emb_size (int): Embedding size.
        num_heads (int): Number of attention heads.
        window_size (int, optional): Window size for attention mechanism. Defaults to 8.
        shifted (bool, optional): If True, use shifted windows. Defaults to True.

        References:
        1. [Swin Transformer GitHub Repository](https://github.com/microsoft/Swin-Transformer)
        2. [Swin Transformer Paper (arXiv:2103.14030)](https://arxiv.org/abs/2103.14030)
        3. [Swin Transformer Blog](https://medium.com/thedeephub/building-swin-transformer-from-scratch-using-pytorch-hierarchical-vision-transformer-using-shifted-91cbf6abc678)
        """
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.window_size = window_size
        self.shifted = shifted

        # Linear layers for projection and output
        self.linear1 = nn.Linear(emb_size, 3 * emb_size)
        self.linear2 = nn.Linear(emb_size, emb_size)

        # Positional embeddings for relative positioning
        self.pos_embeddings = nn.Parameter(torch.randn(window_size * 2 - 1, window_size * 2 - 1))
        # Indices for the relative positional encoding
        self.indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
        # Compute relative indices for the positional encoding
        self.relative_indices = self.indices[None, :, :] - self.indices[:, None, :]
        self.relative_indices += self.window_size - 1

    def forward(self, x):
        """
        Forward pass of the DecoderSelfMSA.

        Parameters:
        x (torch.Tensor): Input tensor of shape (batch_size, num_tokens, emb_size).

        Returns:
        torch.Tensor: Output tensor after multi-head self-attention and projection.
        """
        # Dimension per attention head
        h_dim = self.emb_size / self.num_heads
        # Compute the height and width of the input tensor
        height = width = int(np.sqrt(x.shape[1]))

        # First linear projection to get Q, K, and V
        x = self.linear1(x)

        # Rearrange input for multi-head self-attention
        x = rearrange(x, 'b (h w) (c k) -> b h w c k', h=height, w=width, k=3, c=self.emb_size)
        
        # Apply window shifting if needed
        if self.shifted:
            x = torch.roll(x, (-self.window_size // 2, -self.window_size // 2), dims=(1, 2))
        
        # Further rearrange for multi-head attention
        x = rearrange(x, 'b (Wh w1) (Ww w2) (e H) k -> b H Wh Ww (w1 w2) e k', 
                      w1=self.window_size, w2=self.window_size, H=self.num_heads)            

        # Split into Q, K, and V tensors
        Q, K, V = x.chunk(3, dim=6)
        Q, K, V = Q.squeeze(-1), K.squeeze(-1), V.squeeze(-1)

        # Compute attention weights
        wei = (Q @ K.transpose(4, 5)) / np.sqrt(h_dim)

        # Add relative positional embeddings to attention weights
        rel_pos_embedding = self.pos_embeddings[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        wei += rel_pos_embedding

        # Apply masking for shifted windows
        if self.shifted:
            row_mask = torch.zeros((self.window_size ** 2, self.window_size ** 2)).cuda()
            row_mask[-self.window_size * (self.window_size // 2):, 0:-self.window_size * (self.window_size // 2)] = float('-inf')
            row_mask[0:-self.window_size * (self.window_size // 2), -self.window_size * (self.window_size // 2):] = float('-inf')
            column_mask = rearrange(row_mask, '(r w1) (c w2) -> (w1 r) (w2 c)', w1=self.window_size, w2=self.window_size)
            wei[:, :, -1, :] += row_mask
            wei[:, :, :, -1] += column_mask

        # Apply softmax and multiply with value tensor V
        wei = F.softmax(wei, dim=-1) @ V

        # Rearrange back to original input shape
        x = rearrange(wei, 'b H Wh Ww (w1 w2) e -> b (Wh w1) (Ww w2) (H e)', w1=self.window_size, w2=self.window_size, H=self.num_heads)
        x = rearrange(x, 'b h w c -> b (h w) c')

        # Apply final linear projection
        return self.linear2(x)


class EncoderSelfMSA(nn.Module):
    def __init__(self, emb_size, num_heads, window_size=8, shifted=True):
        """
        Implements a shifted window self-attention mechanism for style, scale and shift features. 
        Uses the same attention map with different projections for style, scale and shift.

        Parameters:
        emb_size (int): Embedding size.
        num_heads (int): Number of attention heads.
        window_size (int, optional): Window size for attention mechanism. Defaults to 8.
        shifted (bool, optional): If True, use shifted windows. Defaults to True.

        References:
        1. [Swin Transformer GitHub Repository](https://github.com/microsoft/Swin-Transformer)
        2. [Swin Transformer Paper (arXiv:2103.14030)](https://arxiv.org/abs/2103.14030)
        3. [Swin Transformer Blog](https://medium.com/thedeephub/building-swin-transformer-from-scratch-using-pytorch-hierarchical-vision-transformer-using-shifted-91cbf6abc678)
        """
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.window_size = window_size
        self.shifted = shifted

        # Linear layers for projection and output
        self.linear1 = nn.Linear(emb_size, 3 * emb_size)
        self.linear1_scale = nn.Linear(emb_size, emb_size)
        self.linear1_shift = nn.Linear(emb_size, emb_size)
        self.linear2 = nn.Linear(emb_size, emb_size)
        self.linear2_scale = nn.Linear(emb_size, emb_size)
        self.linear2_shift = nn.Linear(emb_size, emb_size)

        # Positional embeddings for relative positioning
        self.pos_embeddings = nn.Parameter(torch.randn(window_size * 2 - 1, window_size * 2 - 1))
        # Indices for the relative positional encoding
        self.indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
        # Compute relative indices for the positional encoding
        self.relative_indices = self.indices[None, :, :] - self.indices[:, None, :]
        self.relative_indices += self.window_size - 1

    def forward(self, x, scale, shift):
        """
        Forward pass of the EncoderSelfMSA.

        Parameters:
        x (torch.Tensor): Input tensor of shape (batch_size, num_tokens, emb_size).
        scale (torch.Tensor): Scale tensor of shape (batch_size, num_tokens, emb_size).
        shift (torch.Tensor): Shift tensor of shape (batch_size, num_tokens, emb_size).

        Returns:
        tuple(torch.Tensor, torch.Tensor, torch.Tensor): Output tensor, scale tensor, and shift tensor after multi-head self-attention and affine transformations.
        """
        # Dimension per attention head
        h_dim = self.emb_size / self.num_heads
        # Compute the height and width of the input tensor
        height = width = int(np.sqrt(x.shape[1]))

        # First linear projection to get Q, K, and V
        x = self.linear1(x)
        scale = self.linear1_scale(scale)
        shift = self.linear1_shift(shift)

        # Rearrange input for multi-head self-attention
        x = rearrange(x, 'b (h w) (c k) -> b h w c k', h=height, w=width, k=3, c=self.emb_size)
        scale = rearrange(scale, 'b (h w) (c k) -> b h w c k', h=height, w=width, k=1, c=self.emb_size)
        shift = rearrange(shift, 'b (h w) (c k) -> b h w c k', h=height, w=width, k=1, c=self.emb_size)

        # Apply window shifting if needed
        if self.shifted:
            x = torch.roll(x, (-self.window_size // 2, -self.window_size // 2), dims=(1, 2))
            scale = torch.roll(scale, (-self.window_size // 2, -self.window_size // 2), dims=(1, 2))
            shift = torch.roll(shift, (-self.window_size // 2, -self.window_size // 2), dims=(1, 2))

        # Further rearrange for multi-head attention
        x = rearrange(x, 'b (Wh w1) (Ww w2) (e H) k -> b H Wh Ww (w1 w2) e k',
                      w1=self.window_size, w2=self.window_size, H=self.num_heads)
        scale = rearrange(scale, 'b (Wh w1) (Ww w2) (e H) k -> b H Wh Ww (w1 w2) e k',
                          w1=self.window_size, w2=self.window_size, H=self.num_heads)
        shift = rearrange(shift, 'b (Wh w1) (Ww w2) (e H) k -> b H Wh Ww (w1 w2) e k',
                          w1=self.window_size, w2=self.window_size, H=self.num_heads)

        # Split into Q, K, and V tensors
        Q, K, V = x.chunk(3, dim=6)
        Q, K, V = Q.squeeze(-1), K.squeeze(-1), V.squeeze(-1)

        # Split into V_scale and V_shift tensors
        V_scale = scale.squeeze(-1)
        V_shift = shift.squeeze(-1)

        # Compute attention weights
        wei = (Q @ K.transpose(4, 5)) / np.sqrt(h_dim)

        # Add relative positional embeddings to attention weights
        rel_pos_embedding = self.pos_embeddings[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        wei += rel_pos_embedding

        # Apply masking for shifted windows
        if self.shifted:
            row_mask = torch.zeros((self.window_size ** 2, self.window_size ** 2)).cuda()
            row_mask[-self.window_size * (self.window_size // 2):, 0:-self.window_size * (self.window_size // 2)] = float('-inf')
            row_mask[0:-self.window_size * (self.window_size // 2), -self.window_size * (self.window_size // 2):] = float('-inf')
            column_mask = rearrange(row_mask, '(r w1) (c w2) -> (w1 r) (w2 c)', w1=self.window_size, w2=self.window_size)
            wei[:, :, -1, :] += row_mask
            wei[:, :, :, -1] += column_mask

        # Apply softmax and multiply with value tensor V, V_scale, and V_shift
        w = F.softmax(wei, dim=-1)
        wei = w @ V
        wei_scale = w @ V_scale
        wei_shift = w @ V_shift

        # Rearrange back to original input shape
        x = rearrange(wei, 'b H Wh Ww (w1 w2) e -> b (Wh w1) (Ww w2) (H e)',
                      w1=self.window_size, w2=self.window_size, H=self.num_heads)
        x = rearrange(x, 'b h w c -> b (h w) c')

        scale = rearrange(wei_scale, 'b H Wh Ww (w1 w2) e -> b (Wh w1) (Ww w2) (H e)',
                          w1=self.window_size, w2=self.window_size, H=self.num_heads)
        scale = rearrange(scale, 'b h w c -> b (h w) c')

        shift = rearrange(wei_shift, 'b H Wh Ww (w1 w2) e -> b (Wh w1) (Ww w2) (H e)',
                          w1=self.window_size, w2=self.window_size, H=self.num_heads)
        shift = rearrange(shift, 'b h w c -> b (h w) c')

        # Apply final linear projection
        x = self.linear2(x)
        scale = self.linear2_scale(scale)
        shift = self.linear2_shift(shift)

        return x, scale, shift


class DecoderCrossMSA(nn.Module):
    def __init__(self, emb_size, num_heads, window_size=8, shifted=True):
        """
        Implements a shifted window cross-attention mechanism for scale and shift features. To 
        calculate the attention map uses normaized content and style features. The scale and shift
        features are then computed by multiplying the attention weights with the projected features
        for scale and shift.

        Parameters:
        emb_size (int): Embedding size.
        num_heads (int): Number of attention heads.
        window_size (int, optional): Window size for attention mechanism. Defaults to 8.
        shifted (bool, optional): If True, use shifted windows. Defaults to True.

        References:
        1. [Swin Transformer GitHub Repository](https://github.com/microsoft/Swin-Transformer)
        2. [Swin Transformer Paper (arXiv:2103.14030)](https://arxiv.org/abs/2103.14030)
        3. [Swin Transformer Blog](https://medium.com/thedeephub/building-swin-transformer-from-scratch-using-pytorch-hierarchical-vision-transformer-using-shifted-91cbf6abc678)
        """
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.window_size = window_size
        self.shifted = shifted

        # Linear layers for projection and output
        self.linear1 = nn.Linear(emb_size, emb_size)
        self.linear2 = nn.Linear(emb_size, emb_size)
        self.linear_scale = nn.Linear(emb_size, emb_size)
        self.linear_shift = nn.Linear(emb_size, emb_size)
        self.linear_scale_out = nn.Linear(emb_size, emb_size)
        self.linear_shift_out = nn.Linear(emb_size, emb_size)

        # Positional embeddings for relative positioning
        self.pos_embeddings = nn.Parameter(torch.randn(window_size * 2 - 1, window_size * 2 - 1))
        # Indices for the relative positional encoding
        self.indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
        # Compute relative indices for the positional encoding
        self.relative_indices = self.indices[None, :, :] - self.indices[:, None, :]
        self.relative_indices += self.window_size - 1

    def forward(self, content, style, scale, shift):
        """
        Forward pass of the DecoderCrossMSA.

        Parameters:
        content (torch.Tensor): Input content tensor of shape (batch_size, num_tokens, emb_size).
        style (torch.Tensor): Input style tensor of shape (batch_size, num_tokens, emb_size).
        scale (torch.Tensor): Scale tensor for style transformation of shape (batch_size, num_tokens, emb_size).
        shift (torch.Tensor): Shift tensor for style transformation of shape (batch_size, num_tokens, emb_size).

        Returns:
        tuple(torch.Tensor, torch.Tensor): Scale and shift tensors after multi-head cross-attention and affine transformations.
        """
        # Dimension per attention head
        h_dim = self.emb_size / self.num_heads
        # Compute the height and width of the input tensor
        height = width = int(np.sqrt(content.shape[1]))

        # Linear projection to content, style, scale, and shift
        content = self.linear1(content)
        style = self.linear2(style)
        scale = self.linear_scale(scale)
        shift = self.linear_shift(shift)

        # Rearrange input for multi-head self-attention
        content = rearrange(content, 'b (h w) (c k) -> b h w c k', h=height, w=width, k=1, c=self.emb_size)
        style = rearrange(style, 'b (h w) (c k) -> b h w c k', h=height, w=width, k=1, c=self.emb_size)
        scale = rearrange(scale, 'b (h w) (c k) -> b h w c k', h=height, w=width, k=1, c=self.emb_size)
        shift = rearrange(shift, 'b (h w) (c k) -> b h w c k', h=height, w=width, k=1, c=self.emb_size)

        # Apply window shifting if needed
        if self.shifted:
            content = torch.roll(content, (-self.window_size // 2, -self.window_size // 2), dims=(1, 2))
            style = torch.roll(style, (-self.window_size // 2, -self.window_size // 2), dims=(1, 2))
            scale = torch.roll(scale, (-self.window_size // 2, -self.window_size // 2), dims=(1, 2))
            shift = torch.roll(shift, (-self.window_size // 2, -self.window_size // 2), dims=(1, 2))

        # Further rearrange for multi-head attention
        content = rearrange(content, 'b (Wh w1) (Ww w2) (e H) k -> b H Wh Ww (w1 w2) e k',
                            w1=self.window_size, w2=self.window_size, H=self.num_heads)
        style = rearrange(style, 'b (Wh w1) (Ww w2) (e H) k -> b H Wh Ww (w1 w2) e k',
                          w1=self.window_size, w2=self.window_size, H=self.num_heads)
        scale = rearrange(scale, 'b (Wh w1) (Ww w2) (e H) k -> b H Wh Ww (w1 w2) e k',
                          w1=self.window_size, w2=self.window_size, H=self.num_heads)
        shift = rearrange(shift, 'b (Wh w1) (Ww w2) (e H) k -> b H Wh Ww (w1 w2) e k',
                          w1=self.window_size, w2=self.window_size, H=self.num_heads)

        # Split into Q, K, V_scale, and V_shift tensors
        Q = content.squeeze(-1)
        K = style.squeeze(-1)
        V_scale = scale.squeeze(-1)
        V_shift = shift.squeeze(-1)

        # Compute attention weights
        wei = (Q @ K.transpose(4, 5)) / np.sqrt(h_dim)

        # Add relative positional embeddings to attention weights
        rel_pos_embedding = self.pos_embeddings[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        wei += rel_pos_embedding

        # Apply masking for shifted windows
        if self.shifted:
            row_mask = torch.zeros((self.window_size ** 2, self.window_size ** 2)).cuda()
            row_mask[-self.window_size * (self.window_size // 2):, 0:-self.window_size * (self.window_size // 2)] = float('-inf')
            row_mask[0:-self.window_size * (self.window_size // 2), -self.window_size * (self.window_size // 2):] = float('-inf')
            column_mask = rearrange(row_mask, '(r w1) (c w2) -> (w1 r) (w2 c)', w1=self.window_size, w2=self.window_size)
            wei[:, :, -1, :] += row_mask
            wei[:, :, :, -1] += column_mask

        # Apply softmax and multiply with V_scale and V_shift
        w = F.softmax(wei, dim=-1)
        wei_scale = w @ V_scale
        wei_shift = w @ V_shift

        # Rearrange back to original input shape for scale and shift
        scale = rearrange(wei_scale, 'b H Wh Ww (w1 w2) e -> b (Wh w1) (Ww w2) (H e)',
                          w1=self.window_size, w2=self.window_size, H=self.num_heads)
        scale = rearrange(scale, 'b h w c -> b (h w) c')

        shift = rearrange(wei_shift, 'b H Wh Ww (w1 w2) e -> b (Wh w1) (Ww w2) (H e)',
                          w1=self.window_size, w2=self.window_size, H=self.num_heads)
        shift = rearrange(shift, 'b h w c -> b (h w) c')

        # Apply final linear projections for scale and shift
        return self.linear_scale_out(scale), self.linear_shift_out(shift)


class StyleTransformerEncoder(nn.Module):
    def __init__(self, dim=256, num_heads=8, window_size=8, shifted=True, mlp_ratio=4., act_layer=nn.ReLU):
        """
        Initializes the StyleTransformerEncoder class.

        Parameters:
        dim (int, optional): Dimensionality of the input features. Defaults to 256.
        num_heads (int, optional): Number of attention heads. Defaults to 8.
        window_size (int, optional): Size of the window for attention. Defaults to 8.
        shifted (bool, optional): If True, use shifted windows for attention. Defaults to True.
        mlp_ratio (float, optional): Ratio of hidden dimensions in the MLP layers. Defaults to 4.0.
        act_layer (torch.nn.Module, optional): Activation function to use. Defaults to ReLU.
        """
        super().__init__()
        
        # Multi-head self-attention module
        self.MSA = EncoderSelfMSA(dim, num_heads, window_size, shifted)

        # Determine the hidden dimension for the MLP layers
        mlp_hidden_dim = int(dim * mlp_ratio)

        # MLP layers for key, scale, and shift
        self.MLP_key = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.MLP_scale = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.MLP_shift = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, key, scale, shift):
        """
        Forward pass of the StyleTransformerEncoder.

        Parameters:
        key (torch.Tensor): Input tensor representing the key features.
        scale (torch.Tensor): Input tensor representing the scale features.
        shift (torch.Tensor): Input tensor representing the shift features.

        Returns:
        tuple(torch.Tensor, torch.Tensor, torch.Tensor): Output tensors for key, scale, and shift after self-attention and MLP operations.
        """
        # Save shortcuts for residual connections
        key_shortcut = key
        scale_shortcut = scale
        shift_shortcut = shift

        # Apply multi-head self-attention (MSA)
        key, scale, shift = self.MSA(key, scale, shift)

        # Add the residual connections
        key = key + key_shortcut
        scale = scale + scale_shortcut
        shift = shift + shift_shortcut

        # Apply MLP layers with residual connections
        key = key + self.MLP_key(key)
        scale = scale + self.MLP_scale(scale)
        shift = shift + self.MLP_shift(shift)

        return key, scale, shift


class StyleTransformerDecoder(nn.Module):
    def __init__(self, dim=256, num_heads=8, window_size=8, shifted=True, mlp_ratio=4.0, act_layer=nn.ReLU):
        """
        Initializes the StyleTransformerDecoder class.

        Parameters:
        dim (int, optional): Dimensionality of the input features. Defaults to 256.
        num_heads (int, optional): Number of attention heads. Defaults to 8.
        window_size (int, optional): Size of the window for attention. Defaults to 8.
        shifted (bool, optional): If True, use shifted windows for attention. Defaults to True.
        mlp_ratio (float, optional): Ratio of hidden dimensions in the MLP layers. Defaults to 4.0.
        act_layer (torch.nn.Module, optional): Activation function to use. Defaults to ReLU.
        """
        super().__init__()
        
        # Decoder self-attention (MSA) and cross-attention (CMSA) modules
        self.MSA = DecoderSelfMSA(dim, num_heads, window_size, shifted)
        self.CMSA = DecoderCrossMSA(dim, num_heads, window_size, shifted)

        # Normalization layers for content and style
        self.content_norm = nn.InstanceNorm2d(dim)
        self.style_norm = nn.InstanceNorm2d(dim)

        # Determine the hidden dimension for the MLP layer
        mlp_hidden_dim = int(dim * mlp_ratio)

        # MLP layer for the final projection
        self.MLP_out = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, content, style, scale, shift):
        """
        Forward pass of the StyleTransformerDecoder.

        Parameters:
        content (torch.Tensor): Input tensor representing the content features.
        style (torch.Tensor): Input tensor representing the style features.
        scale (torch.Tensor): Scale tensor for affine transformation.
        shift (torch.Tensor): Shift tensor for affine transformation.

        Returns:
        torch.Tensor: Output tensor after self-attention, cross-attention, and MLP operations.
        """
        # Save shortcut for residual connections
        content_shortcut = content

        # Apply self-attention (MSA)
        content = self.MSA(content)

        # Add the residual connection
        content = content + content_shortcut

        # Normalize content and style
        contentNorm = rearrange(content, 'b l c -> b c l')
        styleNorm = rearrange(style, 'b l c -> b c l')
        contentNorm = self.content_norm(contentNorm)
        styleNorm = self.style_norm(styleNorm)
        contentNorm = rearrange(contentNorm, 'b c l -> b l c')
        styleNorm = rearrange(styleNorm, 'b c l -> b l c')

        # Apply cross-attention (CMSA)
        scale, shift = self.CMSA(contentNorm, styleNorm, scale, shift)

        # Apply affine transformation
        content = content * scale + shift

        # Apply the final MLP layer with residual connection
        content = self.MLP_out(content) + content

        return content
