import torch
import torch.nn as nn

class DepthwiseSeparableBlock_NoSE(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=1):
        super(DepthwiseSeparableBlock_NoSE, self).__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride,
                                padding=1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))

        # 没有 SE 模块

        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

class DSCSELightNet_NoSE(nn.Module):
    def __init__(self, num_classes=10, input_channels=3):
        super(DSCSELightNet_NoSE, self).__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        self.blocks = nn.Sequential(
            DepthwiseSeparableBlock_NoSE(32, 64, stride=2, expand_ratio=1),
            DepthwiseSeparableBlock_NoSE(64, 128, stride=2, expand_ratio=4),
            DepthwiseSeparableBlock_NoSE(128, 128, stride=1, expand_ratio=4),
            DepthwiseSeparableBlock_NoSE(128, 256, stride=2, expand_ratio=4),
            DepthwiseSeparableBlock_NoSE(256, 256, stride=1, expand_ratio=4),
            DepthwiseSeparableBlock_NoSE(256, 512, stride=2, expand_ratio=4),
            DepthwiseSeparableBlock_NoSE(512, 512, stride=1, expand_ratio=4),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x