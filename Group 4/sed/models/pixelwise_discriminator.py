import torch
import torch.nn as nn
from models.semantic_aware_fusion_block import SemanticAwareFusionBlock

class UNetPixelDiscriminator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, num_filters=64):
        super(UNetPixelDiscriminator, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            self._conv_block(in_channels, num_filters),
            self._conv_block(num_filters, num_filters * 2),
            self._conv_block(num_filters * 2, num_filters * 4),
            self._conv_block(num_filters * 4, num_filters * 8),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(num_filters * 8, num_filters * 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            self._upconv_block(num_filters * 8, num_filters * 4),
            self._upconv_block(num_filters * 4, num_filters * 2),
            self._upconv_block(num_filters * 2, num_filters),
            nn.Conv2d(num_filters, out_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def _conv_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def _upconv_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder[0](x)
        enc2 = self.encoder[1](enc1)
        enc3 = self.encoder[2](enc2)
        enc4 = self.encoder[3](enc3)

        # Bottleneck
        bottleneck = self.bottleneck(enc4)

        # Decoder with skip connections using addition
        dec = self.decoder[0](bottleneck)
        dec = self.decoder[1](dec + enc3)
        dec = self.decoder[2](dec + enc2)
        dec = self.decoder[3](dec + enc1)

        return dec


class UNetPixelDiscriminatorwithSed(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, num_filters=64):
        super(UNetPixelDiscriminatorwithSed, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            self._conv_block(in_channels, num_filters),
            self._conv_block(num_filters, num_filters * 2),
            self._conv_block(num_filters * 2, num_filters * 4),
            self._conv_block(num_filters * 4, num_filters * 8),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(num_filters * 8, num_filters * 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            self._upconv_block(num_filters * 8, num_filters * 4),
            self._upconv_block(num_filters * 4, num_filters * 2),
            self._upconv_block(num_filters * 2, num_filters),
            nn.Conv2d(num_filters, out_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

	# Semantic Aware Fusion Blocks
        self.semantic_aware_fusion_block1 = SemanticAwareFusionBlock(channel_size_changer_input_nc=128)
        self.semantic_aware_fusion_block2 = SemanticAwareFusionBlock(channel_size_changer_input_nc=256)
        self.semantic_aware_fusion_block3 = SemanticAwareFusionBlock(channel_size_changer_input_nc=512)

    def _conv_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def _upconv_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, semantic_feature_maps, fs):
        # Encoder 
        enc1 = self.encoder[0](fs) 
        enc2 = self.encoder[1](enc1) 
        enc2 = self.semantic_aware_fusion_block1(semantic_feature_maps, enc2) 
        enc3 = self.encoder[2](enc2) 
        enc3 = self.semantic_aware_fusion_block2(semantic_feature_maps, enc3)
        enc4 = self.encoder[3](enc3) 
        enc4 = self.semantic_aware_fusion_block3(semantic_feature_maps, enc4) 
        # Bottleneck 
        bottleneck = self.bottleneck(enc4) 
        # Decoder with skip connections using addition 
        dec = self.decoder[0](bottleneck) 
        dec = self.decoder[1](dec + enc3) 
        dec = self.decoder[2](dec + enc2) 
        dec = self.decoder[3](dec + enc1) 
        return dec