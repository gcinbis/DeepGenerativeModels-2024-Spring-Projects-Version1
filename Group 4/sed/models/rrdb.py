import torch
import torch.nn as nn

def get_layer(in_channels, out_channels, is_upsample=False, scale_factor=2):
    if is_upsample:
        return nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
    else:
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

def make_blocks(in_channels, out_channels, num_blocks, is_upsample=False):
    blocks = []
    for i in range(num_blocks):
        conv_block = get_layer(
            in_channels + out_channels * i,
            out_channels if i < num_blocks - 1 else in_channels,
            is_upsample,
        )
        blocks.append(conv_block)
    return nn.ModuleList(blocks)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=5, is_upsample=False):
        super().__init__()
        self.blocks = make_blocks(in_channels, out_channels, num_blocks, is_upsample)

    def forward(self, x):
        prev_features = x
        for block in self.blocks:
            current_output = block(prev_features)
            prev_features = torch.cat([prev_features, current_output], dim=1)
        return x + current_output * 0.2

class Residual_in_ResidualBlock(nn.Module):
    def __init__(self, in_channels, num_blocks=3, is_upsample=False):
        super().__init__()
        self.rrdb1 = DenseBlock(in_channels, in_channels, num_blocks, is_upsample)
        self.rrdb2 = DenseBlock(in_channels, in_channels, num_blocks, is_upsample)
        self.rrdb3 = DenseBlock(in_channels, in_channels, num_blocks, is_upsample)
        
    def forward(self, x):
        out1 = self.rrdb1(x)
        out2 = self.rrdb2(out1)
        out3 = self.rrdb3(out2)
        return x + out3 * 0.2



class RRDBNet(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_blocks=23):
        super().__init__()
        self.conv1= get_layer(in_channels, num_channels)
        self.conv2 = get_layer(num_channels, num_channels)
        self.conv3 =   get_layer(num_channels, num_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.output = get_layer(num_channels, in_channels)

        self.first_ups = get_layer(num_channels, num_channels, is_upsample=True)
        self.second_ups = get_layer(num_channels, num_channels, is_upsample=True)

        self.rrdb= nn.Sequential(*[Residual_in_ResidualBlock(num_channels) for _ in range(num_blocks)])


    def forward(self, x):
        res = self.conv1(x)
        x = self.rrdb(res)
        x = self.conv2(x)
        x = x + res
        x = self.first_ups(x)
        x = self.second_ups(x)
        x = self.act(self.conv3(x))
        x = self.output(x)
        return x