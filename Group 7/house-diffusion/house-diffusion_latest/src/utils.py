import torch

def dec2bin(decimal_input, num_bits):
    """
    Converts a decimal number to its binary representation.
    Ref: https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits

    Args:
        decimal_input (torch.Tensor): The decimal number to be converted.
        num_bits (int): The number of bits in the binary representation.

    Returns:
        torch.Tensor: The binary representation of the decimal number.
    """
    # Check if the input is float
    if decimal_input.dtype == torch.float32:
        decimal_input = decimal_input.round().int()
    mask = 2 ** torch.arange(num_bits - 1, -1, -1).to(decimal_input.device, decimal_input.dtype)
    binary_output = decimal_input.unsqueeze(-1).bitwise_and(mask).ne(0).float()
    return binary_output

def broadcast_into(x, timesteps, target_shape):
    """
    Broadcast a tensor into a specified shape.

    Args:
        x (torch.Tensor): The tensor to broadcast.
        timesteps (torch.Tensor): The timesteps to broadcast over.
        shape (tuple): The shape to broadcast into.

    Returns:
        torch.Tensor: The broadcasted tensor.
    """
    return x[timesteps].view(-1, *(1,) * (len(target_shape) - 1)).expand(target_shape)