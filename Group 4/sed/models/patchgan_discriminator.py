import torch
import sys
import torch.nn as nn
sys.path.append('/scratch/users/hpc-yekin/hpc_run/SeD/models')
from semantic_aware_fusion_block import SemanticAwareFusionBlock

class PatchDiscriminator(nn.Module):
    def __init__(self, input_channels, num_filters=64):
        super().__init__()
        self.downsampler = DownSampler(input_channels, num_filters)
        self.final_conv = nn.Conv2d(num_filters * 8, 1, kernel_size=4, stride=1, padding=1)
        
    def forward(self, fs):
        fs = self.downsampler(fs)
        fs = self.final_conv(fs)
        return fs
    
class DownSampler(nn.Module):
    #downsamples 4 times in a conv, bn, leaky relu fashion that halves the spatial dimensions in each step and doubles the number of filters
    def __init__(self, input_channels, num_filters=64):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_filters, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        self.conv2 = nn.Conv2d(num_filters, num_filters * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters * 2)
        
        self.conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(num_filters * 4)
        
        self.conv4 = nn.Conv2d(num_filters * 4, num_filters * 8, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(num_filters * 8)
        
    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        return x
    
    
class PatchDiscriminatorWithSeD(nn.Module):
    def __init__(self, input_channels, num_filters=64):
        super().__init__()
        self.downsampler = DownSampler(input_channels, num_filters)
        self.semantic_aware_fusion_block1 = SemanticAwareFusionBlock()
        self.semantic_aware_fusion_block2 = SemanticAwareFusionBlock(channel_size_changer_input_nc=1024)
        self.semantic_aware_fusion_block3 = SemanticAwareFusionBlock(channel_size_changer_input_nc=1024)
        self.final_conv = nn.Conv2d(num_filters * 16, 1, kernel_size=4, stride=1, padding=1)
        
    def forward(self, semantic_feature_maps, fs):
        x = self.downsampler(fs)
        x = self.semantic_aware_fusion_block1(semantic_feature_maps, x)
        x = self.semantic_aware_fusion_block2(semantic_feature_maps, x)
        x = self.semantic_aware_fusion_block3(semantic_feature_maps, x)
        x = self.final_conv(x)
        return x
    

    
if __name__ == "__main__":
    b = torch.randn(1, 3, 256, 256)
    a = torch.randn(1,1024,16,16)
    model = PatchDiscriminator(3)
    with torch.no_grad():
        output = model(b)