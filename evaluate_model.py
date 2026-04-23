import torch
import time
from thop import profile
from model import DSCSELightNet


def compute_flops_and_params(model, input_size=(1, 3, 32, 32)):
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size).to(device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    return flops, params


def measure_inference_time(model, batch_size=64, num_runs=100, warmup=10):
    device = next(model.parameters()).device
    dummy_input = torch.randn(batch_size, 3, 32, 32).to(device)
    model.eval()

    # Warmup
    for _ in range(warmup):
        _ = model(dummy_input)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_runs):
        _ = model(dummy_input)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - start
    avg_time_ms = (elapsed / num_runs) * 1000
    return avg_time_ms


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 原始模型
    print("\n=== Original Model (best_model.pth) ===")
    model_orig = DSCSELightNet(num_classes=10).to(device)
    model_orig.load_state_dict(torch.load('best_model.pth', map_location=device))
    flops, params = compute_flops_and_params(model_orig)
    print(f"FLOPs: {flops / 1e6:.2f} M")
    print(f"Params: {params / 1e6:.2f} M")
    time_ms = measure_inference_time(model_orig, batch_size=64)
    print(f"Inference time (batch=64): {time_ms:.2f} ms")

    # 2. 剪枝+微调后的模型（finetuned_pruned_model.pth 是 state_dict，需要先加载结构）
    print("\n=== Pruned 30% + Fine-tuned Model ===")
    # 先加载剪枝后的完整模型结构
    model_pruned_full = torch.load('real_pruned_model_30.pth', weights_only=False)
    # 加载微调后的权重
    finetuned_state = torch.load('finetuned_pruned_model.pth', map_location=device)
    # 修复分类头（如果输出不是10）
    if hasattr(model_pruned_full, 'classifier'):
        last = model_pruned_full.classifier[-1]
        if isinstance(last, torch.nn.Linear) and last.out_features != 10:
            in_features = last.in_features
            model_pruned_full.classifier[-1] = torch.nn.Linear(in_features, 10).to(device)
    model_pruned_full.load_state_dict(finetuned_state, strict=False)
    model_pruned_full = model_pruned_full.to(device)
    flops2, params2 = compute_flops_and_params(model_pruned_full)
    print(f"FLOPs: {flops2 / 1e6:.2f} M")
    print(f"Params: {params2 / 1e6:.2f} M")
    time_ms2 = measure_inference_time(model_pruned_full, batch_size=64)
    print(f"Inference time (batch=64): {time_ms2:.2f} ms")

    # 3. 蒸馏后的模型（distilled_student_full.pth）
    print("\n=== Distilled Student Model ===")
    model_distilled = torch.load('distilled_student_full.pth', weights_only=False)
    model_distilled = model_distilled.to(device)
    flops3, params3 = compute_flops_and_params(model_distilled)
    print(f"FLOPs: {flops3 / 1e6:.2f} M")
    print(f"Params: {params3 / 1e6:.2f} M")
    time_ms3 = measure_inference_time(model_distilled, batch_size=64)
    print(f"Inference time (batch=64): {time_ms3:.2f} ms")

    # 打印对比表
    print("\n=== Summary Table ===")
    print(f"{'Model':<25} {'FLOPs (M)':<12} {'Params (M)':<12} {'Inference (ms/batch)':<20}")
    print("-" * 70)
    print(f"{'Original':<25} {flops / 1e6:<12.2f} {params / 1e6:<12.2f} {time_ms:<20.2f}")
    print(f"{'Pruned 30% + FT':<25} {flops2 / 1e6:<12.2f} {params2 / 1e6:<12.2f} {time_ms2:<20.2f}")
    print(f"{'Distilled':<25} {flops3 / 1e6:<12.2f} {params3 / 1e6:<12.2f} {time_ms3:<20.2f}")


if __name__ == '__main__':
    main()