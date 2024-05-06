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
    mask = 2 ** torch.arange(num_bits - 1, -1, -1).to(decimal_input.device, decimal_input.dtype)
    binary_output = decimal_input.unsqueeze(-1).bitwise_and(mask).ne(0).float()
    return binary_output