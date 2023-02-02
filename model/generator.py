import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        return torch.add(out, identity)


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

class Generator(nn.Module):
    """Generator of SRGAN.

    Args:
        scale_factor (int): scale factor of target
        B (int): number of residual blocks
    """
    def __init__(self, scale_factor: int = 2, B: int = 16) -> None:
        super(Generator, self).__init__()
        upsample_block_num = int(math.log(scale_factor, 2))
        
        # first layer
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )
        
        # residual blocks
        residual_blocks = [ResidualBlock(64) for _ in range(B)]
        self.residual_blocks = nn.Sequential(*residual_blocks)
        
        # second conv layer after residual blocks
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64)
        )
        
        upscale_block = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        self.upscale_block = nn.Sequential(*upscale_block)
        
        # final output layer
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4),
            nn.Tanh()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.block1(x)
        out2 = self.residual_blocks(out1)
        out3 = self.block2(out2)
        out3 = torch.add(out3, out1)
        out4 = self.upscale_block(out3)
        out5 = self.block3(out4)
        
        return out5


if __name__ == "__main__":
    from torchsummary import summary
    model = Generator()
    summary(model, (3, 224, 224), device="cpu")