import torch
from thop import profile
from model_without_se import DSCSELightNet_NoSE
from model_without_dsc import DSCSELightNet_NoDSC

def compute(model, name, acc=None):
    input_tensor = torch.randn(1, 3, 32, 32)
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    print(f"{name:20} | FLOPs: {flops/1e6:8.2f} M | Params: {params/1e6:6.2f} M | Acc: {acc if acc else 'N/A'}%")
    return flops/1e6, params/1e6

if __name__ == '__main__':
    model_nose = DSCSELightNet_NoSE(num_classes=10)
    model_nodsc = DSCSELightNet_NoDSC(num_classes=10)

    # 准确率从你的日志中获取
    acc_nose = 89.80   # 你训练得到的准确率
    acc_nodsc = None   # 暂时未知，可以填空

    compute(model_nose, "No SE", acc_nose)
    compute(model_nodsc, "No DSC", acc_nodsc)