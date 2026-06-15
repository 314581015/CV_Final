import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel attention: learns which channels to focus on."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        avg = self.fc(x.mean(dim=[2, 3]))
        mx = self.fc(x.amax(dim=[2, 3]))
        weight = torch.sigmoid(avg + mx).view(b, c, 1, 1)
        return x * weight


class SpatialAttention(nn.Module):
    """Spatial attention: learns where (which positions) to focus on."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        weight = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * weight


class CBAM(nn.Module):
    """Convolutional Block Attention Module: channel then spatial attention.

    YAML usage:  - [-1, 1, CBAM, [256, 16, 7]]
    """

    def __init__(self, c1: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(c1, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


class ECA(nn.Module):
    """Efficient Channel Attention: lightweight channel attention via 1D conv.

    YAML usage:  - [-1, 1, ECA, [3]]
    """

    def __init__(self, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)                              # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)               # [B, 1, C]
        y = self.conv(y)                                  # [B, 1, C]
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        return x * y.expand_as(x)


class CoordAtt(nn.Module):
    """Coordinate Attention: preserves spatial coordinate info for small objects.

    YAML usage:  - [-1, 1, CoordAtt, [128]]
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mip = max(8, channels // 16)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        x_h = self.pool_h(x)                             # [B, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)         # [B, C, W, 1]
        y = torch.cat([x_h, x_w], dim=2)                 # [B, C, H+W, 1]
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))
        return x * a_h * a_w
