# 基于轻量化深度卷积网络的图像识别模型优化研究

本项目论文《基于轻量化深度卷积网络的图像识别模型优化研究》（DSC-SE-LightNet），实现了深度可分离卷积、SE注意力机制、通道剪枝、知识蒸馏和8-bit量化等优化策略，并在CIFAR-10数据集上进行了验证。

## 主要结果

| 模型 | 参数量 (M) | FLOPs (M) | Top-1 准确率 (%) |
|------|-----------|-----------|---------------|
| 原始模型 | 4.73 | 48.64     | 90.53         |
| 30%剪枝+微调 | 2.32 | 24.36     | 88.73         |
| 30%剪枝+蒸馏 | 2.32 | 24.36     | 89.95         |
| 50%剪枝+微调 | 1.20 | ~12.6     | 85.34         |

消融实验证实SE模块贡献约0.7%精度，深度可分离卷积可大幅降低计算量。

## 环境配置

- Python 3.9+
- PyTorch 1.10+
- torchvision, thop, tqdm, matplotlib, pandas

安装依赖：

```bash
pip install torch torchvision thop tqdm matplotlib pandas

快速开始
1. 训练原始模型
bash
python train.py          # 200轮，约1-2小时
2. 通道剪枝（30%）
bash
python real_prune.py     # 生成 real_pruned_model_30.pth
python finetune.py       # 微调20轮
python distillation.py   # 知识蒸馏30轮（可选）
3. 评估指标
bash
python evaluate_model.py   # 输出FLOPs、参数量、推理时间
python generate_all_charts.py  # 生成训练曲线、柱状图、散点图
4. 消融实验
bash
python train_no_se.py      # 训练无SE模型（已提供训练脚本）
python compute_stats_no_weights.py  # 计算无SE/无DSC的FLOPs
目录结构
model.py – 核心模型定义

train.py – 原始模型训练

real_prune.py – 剪枝

finetune.py – 微调

distillation.py – 知识蒸馏

evaluate_model.py – 性能评估

generate_all_charts.py – 图表生成

*.png – 实验结果图

*.docx – 结果表格（可直接用于论文）

注意事项
数据集CIFAR-10会在首次运行时自动下载。

模型权重文件（.pth）未上传至仓库，请自行运行训练脚本生成，或从网盘下载（见Releases）。

若需在CPU上运行，将代码中device = torch.device("cuda")改为"cpu"。

引用
若本代码对您有帮助，请引用原论文（见项目主页）。

许可
MIT License