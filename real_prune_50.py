import torch_pruning as tp
import torch
from model import DSCSELightNet

model = DSCSELightNet(num_classes=10)
model.load_state_dict(torch.load('best_model.pth'))

importance = tp.importance.MagnitudeImportance(p=2)

pruner = tp.pruner.MetaPruner(
    model,
    example_inputs=torch.randn(1, 3, 32, 32),
    importance=importance,
    iterative_steps=1,
    pruning_ratio=0.5,   # 50% 剪枝
    ignored_layers=[],
)

pruner.step()

total_params = sum(p.numel() for p in model.parameters())
print(f"剪枝后参数量: {total_params / 1e6:.2f} M")

torch.save(model, 'real_pruned_model_50.pth')
print("剪枝后的完整模型已保存为 real_pruned_model_50.pth")