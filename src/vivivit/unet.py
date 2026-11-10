import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.checkpoint as checkpoint

class OptimizedConvBlock3D(nn.Module):

    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        if num_groups > out_channels:
            num_groups = out_channels
        if out_channels % num_groups != 0:
            g = out_channels
            while g > 1 and out_channels % g != 0:
                g //= 2
            num_groups = g if g > 1 else 1
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += identity
        return self.relu(out)

class UNet3D(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, base_channels=32, num_levels=4, use_checkpoint_cnn=False, num_groups=8):
        super().__init__()
        self.use_checkpoint_cnn = use_checkpoint_cnn
        channels = [base_channels * 2 ** i for i in range(num_levels + 1)]
        self.init_conv = OptimizedConvBlock3D(in_channels, channels[0], num_groups=num_groups)
        self.encoder_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(num_levels):
            self.encoder_blocks.append(OptimizedConvBlock3D(channels[i], channels[i + 1], num_groups=num_groups))
            self.pools.append(nn.MaxPool3d(2))
        self.bottleneck = OptimizedConvBlock3D(channels[num_levels], channels[num_levels], num_groups=num_groups)
        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for i in range(num_levels - 1, -1, -1):
            self.upsamples.append(nn.ConvTranspose3d(channels[i + 1], channels[i], 2, 2))
            self.decoder_blocks.append(OptimizedConvBlock3D(channels[i] * 2, channels[i], num_groups=num_groups))
        self.final_conv = nn.Conv3d(channels[0], out_channels, 1)

    def forward(self, x):
        skips = []
        out = self.init_conv(x)
        for (block, pool) in zip(self.encoder_blocks, self.pools):
            skips.append(out)
            if self.training and self.use_checkpoint_cnn:
                out = checkpoint.checkpoint(block, out, use_reentrant=False)
            else:
                out = block(out)
            out = pool(out)
        out = self.bottleneck(out)
        for (i, (up, block)) in enumerate(zip(self.upsamples, self.decoder_blocks)):
            out = up(out)
            skip = skips[-(i + 1)]
            if out.shape != skip.shape:
                out = F.interpolate(out, size=skip.shape[2:], mode='trilinear', align_corners=False)
            out = torch.cat([out, skip], dim=1)
            if self.training and self.use_checkpoint_cnn:
                out = checkpoint.checkpoint(block, out, use_reentrant=False)
            else:
                out = block(out)
        return self.final_conv(out)
