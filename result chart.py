import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

models = ['完整模型', '无DSC(估)', '无SE', '无剪枝', '无蒸馏']
acc = [89.95, 90.5, 89.80, 90.53, 88.73]
latency = [5.34, 6.01, 5.8, 6.01, 6.18]

plt.figure(figsize=(8,6))
plt.scatter(latency, acc, s=100)
for i, model in enumerate(models):
    plt.annotate(model, (latency[i], acc[i]))
plt.xlabel('推理时间 (ms/batch)')
plt.ylabel('Top-1 准确率 (%)')
plt.title('消融实验：准确率 vs 推理时间')
plt.grid(True)
plt.savefig('ablation_scatter_cn.png', dpi=150)
plt.show()