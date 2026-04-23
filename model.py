import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        # 全局平均池化，输出形状 [batch, channels, 1, 1]
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # 全连接层1：降维
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        # 全连接层2：升维回原通道数
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: 全局平均池化，输出 [b, c]
        y = self.global_pool(x).view(b, c)
        # Excitation: 学习通道权重
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        # 缩放原特征
        return x * y.expand_as(x)

class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=1, use_se=True):
        super(DepthwiseSeparableBlock, self).__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)

        layers = []
        # 1. 升维（如果 expand_ratio > 1）
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        # 2. Depthwise Conv
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride,
                                padding=1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))

        # 3. SE 模块（可选）
        if use_se:
            layers.append(SEBlock(hidden_dim))

        # 4. Pointwise Conv（降维）
        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

class DSCSELightNet(nn.Module):
    def __init__(self, num_classes=10, input_channels=3):
        super(DSCSELightNet, self).__init__()

        # 第一层标准卷积
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        # 定义深度可分离块序列 (in, out, stride, expand_ratio)
        self.blocks = nn.Sequential(
            DepthwiseSeparableBlock(32, 64, stride=2, expand_ratio=1, use_se=True),
            DepthwiseSeparableBlock(64, 128, stride=2, expand_ratio=4, use_se=True),
            DepthwiseSeparableBlock(128, 128, stride=1, expand_ratio=4, use_se=True),
            DepthwiseSeparableBlock(128, 256, stride=2, expand_ratio=4, use_se=True),
            DepthwiseSeparableBlock(256, 256, stride=1, expand_ratio=4, use_se=True),
            DepthwiseSeparableBlock(256, 512, stride=2, expand_ratio=4, use_se=True),
            DepthwiseSeparableBlock(512, 512, stride=1, expand_ratio=4, use_se=True),
        )

        # 全局平均池化 + 分类头
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

if __name__ == "__main__":
    model = DSCSELightNet(num_classes=10)
    x = torch.randn(1, 3, 32, 32)  # 模拟 CIFAR-10 输入
    y = model(x)
    print("输出形状:", y.shape)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params / 1e6:.2f} M")