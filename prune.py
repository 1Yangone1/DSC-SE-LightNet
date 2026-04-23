import torch
import torch.nn.utils.prune as prune
from model import DSCSELightNet


def prune_model(model, pruning_rate=0.3):
    """
    对模型的所有深度可分离卷积层进行通道剪枝
    pruning_rate: 剪枝比例，例如 0.3 表示剪掉 30% 的通道
    """
    # 1. 为什么只剪枝卷积层？
    #    因为全连接层参数量小，剪枝收益不大。论文主要对卷积层剪枝。
    for name, module in model.named_modules():
        # 只剪枝 Conv2d 层，并且是 depthwise 卷积（groups == in_channels）
        if isinstance(module, torch.nn.Conv2d) and module.groups == module.in_channels:
            # 2. 为什么用 L1 范数剪枝？
            #    论文中提到“权重范数”作为重要性指标，L1 范数简单有效。
            prune.l1_unstructured(module, name='weight', amount=pruning_rate)
            # 3. 为什么需要 remove？
            #    prune 只是添加了掩码，remove 后才真正将剪枝后的权重保存下来。
            prune.remove(module, 'weight')
    return model


if __name__ == '__main__':
    # 加载之前训练好的最佳模型
    model = DSCSELightNet(num_classes=10)
    model.load_state_dict(torch.load('best_model.pth'))

    # 剪枝前参数量
    total_params_before = sum(p.numel() for p in model.parameters())
    print(f"剪枝前参数量: {total_params_before / 1e6:.2f} M")

    # 执行 30% 剪枝
    model = prune_model(model, pruning_rate=0.3)

    # 剪枝后参数量（注意：PyTorch 的 prune.remove 不会真正减少参数量，只是将权重中不重要的元素置零）
    # 为了真实测量剪枝后的参数量，需要手动重构模型。这里先演示流程。
    total_params_after = sum(p.numel() for p in model.parameters())
    print(f"剪枝后参数量（掩码置零，未删除）: {total_params_after / 1e6:.2f} M")

    # 保存剪枝后的模型
    torch.save(model.state_dict(), 'pruned_model_30.pth')
    print("剪枝完成，模型已保存为 pruned_model_30.pth")