import torch_pruning as tp
import torch
from model import DSCSELightNet

# 加载原始训练好的模型
model = DSCSELightNet(num_classes=10)
model.load_state_dict(torch.load('best_model.pth'))

# 定义重要性评估函数
importance = tp.importance.MagnitudeImportance(p=2)

# 剪枝配置
pruner = tp.pruner.MetaPruner(
    model,
    example_inputs=torch.randn(1, 3, 32, 32),
    importance=importance,
    iterative_steps=1,
    pruning_ratio=0.3,  # 剪枝30%
    ignored_layers=[],
)

# 执行剪枝
pruner.step()

# 查看剪枝后参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"剪枝后参数量: {total_params / 1e6:.2f} M")

# 保存整个模型（包括结构）
torch.save(model, 'real_pruned_model_30.pth')
print("剪枝后的完整模型已保存为 real_pruned_model_30.pth")