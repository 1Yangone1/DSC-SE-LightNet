import torch
import torch.quantization
from model import DSCSELightNet

# 加载原始模型
model = DSCSELightNet(num_classes=10)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# 动态量化（对 Linear 和 Conv1d/Conv2d 层）
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
)

# 保存量化后的模型
torch.save(quantized_model.state_dict(), 'quantized_model.pth')

# 计算体积
def get_model_size(state_dict):
    import io
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return buffer.tell() / 1024  # KB

orig_size = get_model_size(torch.load('best_model.pth', map_location='cpu'))
quant_size = get_model_size(torch.load('quantized_model.pth', map_location='cpu'))
print(f"Original model size: {orig_size:.2f} KB")
print(f"Quantized model size: {quant_size:.2f} KB")
print(f"Compression ratio: {orig_size/quant_size:.2f}x")

# 可选：评估量化后的准确率（需要测试集）
# 注意：动态量化对准确率影响很小