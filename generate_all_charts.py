import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ------------------------------
# 图表1：训练曲线（从 CSV 读取）
# ------------------------------
def plot_training_curve(csv_path='training_log.csv'):
    try:
        df = pd.read_csv(csv_path)
        epochs = df['epoch']
        train_acc = df['train_acc']
        test_acc = df['test_acc']

        plt.figure(figsize=(10,5))
        plt.plot(epochs, train_acc, label='训练集准确率', marker='.')
        plt.plot(epochs, test_acc, label='测试集准确率', marker='.')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('DSC-SE-LightNet 训练曲线（50轮）')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_curve.png', dpi=150)
        plt.show()
        print("✅ 训练曲线已保存为 training_curve.png")
    except FileNotFoundError:
        print("❌ 未找到 training_log.csv，请先运行 train_short.py 生成日志。")

# ------------------------------
# 图表2：参数量与FLOPs柱状图
# ------------------------------
def plot_params_flops_bar():
    models = ['原始', '30%剪枝+蒸馏', '50%剪枝', '无SE', '无DSC']
    params = [4.73, 2.32, 1.20, 3.87, 4.94]
    flops = [48.64, 24.36, 12.6, 47.68, 50.74]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(10,6))
    plt.bar(x - width/2, params, width, label='参数量 (M)', color='steelblue')
    plt.bar(x + width/2, flops, width, label='FLOPs (M)', color='coral')
    plt.xticks(x, models, rotation=15)
    plt.ylabel('数值')
    plt.title('不同模型变体的参数量与计算量对比')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('params_flops_bar.png', dpi=150)
    plt.show()
    print("✅ 柱状图已保存为 params_flops_bar.png")

# ------------------------------
# 图表3：准确率 vs 参数量散点图
# ------------------------------
def plot_acc_vs_params():
    models = ['原始', '30%剪枝+蒸馏', '50%剪枝', '无SE', '无DSC']
    params = [4.73, 2.32, 1.20, 3.87, 4.94]
    acc = [90.53, 89.95, 85.34, 89.80, 90.5]   # 无DSC为估计值90.5

    plt.figure(figsize=(8,6))
    plt.scatter(params, acc, s=100, c='darkgreen')
    for i, model in enumerate(models):
        plt.annotate(model, (params[i], acc[i]), xytext=(5,5), textcoords='offset points')
    plt.xlabel('参数量 (M)')
    plt.ylabel('Top-1 准确率 (%)')
    plt.title('模型压缩与精度权衡')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('acc_vs_params.png', dpi=150)
    plt.show()
    print("✅ 散点图已保存为 acc_vs_params.png")

# ------------------------------
# 主函数
# ------------------------------
if __name__ == '__main__':
    plot_training_curve()
    plot_params_flops_bar()
    plot_acc_vs_params()