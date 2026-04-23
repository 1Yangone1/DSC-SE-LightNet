import torch
import torch.nn as nn
from model import SEBlock  # 复用SE模块

class StandardBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super(StandardBlock, self).__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(out_channels)
        if self.use_residual:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.use_se:
            out = self.se(out)
        identity = self.shortcut(x)
        out = out + identity
        return out

class DSCSELightNet_NoDSC(nn.Module):
    def __init__(self, num_classes=10, input_channels=3):
        super(DSCSELightNet_NoDSC, self).__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        self.blocks = nn.Sequential(
            StandardBlock(32, 64, stride=2, use_se=True),
            StandardBlock(64, 128, stride=2, use_se=True),
            StandardBlock(128, 128, stride=1, use_se=True),
            StandardBlock(128, 256, stride=2, use_se=True),
            StandardBlock(256, 256, stride=1, use_se=True),
            StandardBlock(256, 512, stride=2, use_se=True),
            StandardBlock(512, 512, stride=1, use_se=True),
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