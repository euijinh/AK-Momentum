"""
CIFAR-10 architectures used in the paper's generality checks.

    mlp        3 hidden layers (4096, 4096, 2048), LayerNorm + GELU + dropout 0.5,
               classifier named ``fc``.  This is the AK-SGD vs SGD-momentum test.
    resnet18   standard BasicBlock ResNet-18 with the CIFAR stem (3x3, stride 1,
               no max-pool), so 32x32 inputs are not down-sampled too early.
    vit_tiny   ViT-Ti (dim 192, depth 12, 3 heads) with patch 4 on 32x32 inputs,
               giving an 8x8 = 64-token grid plus the class token.

The head is named ``fc`` in all three, and the ViT keeps ``cls_token`` 1-D and
``pos_embed`` outside the captured modules, so that parameter grouping by name
behaves consistently across architectures.
"""

import math
from typing import List, Optional, Sequence, Type

import torch
import torch.nn as nn

__all__ = ["build_model", "CIFAR10MLP", "ResNet", "VisionTransformer"]


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class CIFAR10MLP(nn.Module):
    """Deep MLP over flattened CIFAR-10 images."""

    def __init__(self, num_classes: int = 10,
                 hidden_dims: Sequence[int] = (4096, 4096, 2048),
                 dropout: float = 0.5):
        super().__init__()
        blocks: List[nn.Module] = []
        prev = 32 * 32 * 3
        for h in hidden_dims:
            blocks += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        self.body = nn.Sequential(*blocks)
        self.fc = nn.Linear(prev, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.body(torch.flatten(x, 1)))


# ---------------------------------------------------------------------------
# ResNet-18
# ---------------------------------------------------------------------------

def conv3x3(cin: int, cout: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        identity = x if self.downsample is None else self.downsample(x)
        return self.relu(out + identity)


class ResNet(nn.Module):
    """ResNet with the CIFAR stem; ``layers`` gives the blocks per stage."""

    def __init__(self, block: Type[BasicBlock], layers: Sequence[int], num_classes: int = 10):
        super().__init__()
        self.inplanes = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        layers += [block(self.inplanes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.avgpool(x), 1))


# ---------------------------------------------------------------------------
# ViT-Tiny
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Patchify with a stride-`patch_size` convolution."""

    def __init__(self, img_size: int, patch_size: int, in_chans: int, dim: int):
        super().__init__()
        if img_size % patch_size:
            raise ValueError(f"img_size {img_size} must be divisible by patch {patch_size}")
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(b, n, c))


class MLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """Pre-norm encoder block."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLPBlock(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class VisionTransformer(nn.Module):
    def __init__(self, img_size: int = 32, patch_size: int = 4, num_classes: int = 10,
                 dim: int = 192, depth: int = 12, num_heads: int = 3, mlp_ratio: float = 4.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, dim)
        # 1-D class token and the position table stay outside the captured
        # weights, so both fall back to the base optimizer's first moment.
        self.cls_token = nn.Parameter(torch.zeros(dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, dim))
        self.blocks = nn.ModuleList([Block(dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                nn.init.trunc_normal_(m.weight, std=math.sqrt(1.0 / fan_in))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], 1, -1)
        x = torch.cat((cls, x), dim=1) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.fc(self.norm(x)[:, 0])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def build_model(name: str, num_classes: int = 10) -> nn.Module:
    """Instantiate ``'mlp'``, ``'resnet18'`` or ``'vit_tiny'``."""
    key = name.lower()
    if key == "mlp":
        return CIFAR10MLP(num_classes=num_classes)
    if key == "resnet18":
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    if key == "vit_tiny":
        return VisionTransformer(num_classes=num_classes)
    raise ValueError(f"Unknown model '{name}'. Choose from: mlp, resnet18, vit_tiny.")
