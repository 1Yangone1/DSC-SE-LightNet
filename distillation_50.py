import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import DSCSELightNet


def kd_loss(student_output, teacher_output, target, T=4.0, alpha=0.7):
    soft_student = nn.functional.log_softmax(student_output / T, dim=1)
    soft_teacher = nn.functional.softmax(teacher_output / T, dim=1)
    soft_loss = nn.functional.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T * T)
    hard_loss = nn.functional.cross_entropy(student_output, target)
    return alpha * soft_loss + (1 - alpha) * hard_loss


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total


def fix_classifier_head(model, num_classes=10):
    """修复模型分类头，确保输出维度为 num_classes"""
    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
        last_layer = model.classifier[-1]
        if isinstance(last_layer, nn.Linear) and last_layer.out_features != num_classes:
            print(f"Fixing classifier: from {last_layer.out_features} to {num_classes}")
            in_features = last_layer.in_features
            new_layer = nn.Linear(in_features, num_classes)
            # 将新层放到相同设备
            device = next(model.parameters()).device
            new_layer = new_layer.to(device)
            model.classifier[-1] = new_layer
    return model


def main():
    BATCH_SIZE = 64
    EPOCHS = 30
    LEARNING_RATE = 0.0001
    T = 4.0
    ALPHA = 0.7
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", DEVICE)

    # 数据加载
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 教师模型
    teacher = DSCSELightNet(num_classes=10).to(DEVICE)
    teacher.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
    teacher.eval()
    print("Loaded teacher model")

    # 学生模型：加载剪枝后的完整模型
    student = torch.load('real_pruned_model_50.pth', weights_only=False)
    student = fix_classifier_head(student, 10)  # 同样需要修复函数
    finetuned_state = torch.load('finetuned_pruned_model_50.pth', map_location=DEVICE)

    # 修复分类头为10类
    student = fix_classifier_head(student, 10)

    # 加载微调后的权重（state_dict），仅加载匹配的层
    finetuned_state = torch.load('finetuned_pruned_model.pth', map_location=DEVICE)
    model_dict = student.state_dict()
    matched_state = {k: v for k, v in finetuned_state.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(matched_state)
    student.load_state_dict(model_dict)
    print("Loaded fine-tuned weights into student")

    # 评估蒸馏前准确率
    acc_before = evaluate(student, testloader, DEVICE)
    print(f"Student accuracy before distillation: {acc_before:.2f}%")

    optimizer = optim.Adam(student.parameters(), lr=LEARNING_RATE)

    best_acc = acc_before
    for epoch in range(EPOCHS):
        student.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(trainloader, desc=f"Distillation Epoch {epoch + 1}/{EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.no_grad():
                teacher_output = teacher(images)
            student_output = student(images)
            loss = kd_loss(student_output, teacher_output, labels, T=T, alpha=ALPHA)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = student_output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix(loss=running_loss / (total / BATCH_SIZE), acc=100. * correct / total)

        acc = evaluate(student, testloader, DEVICE)
        print(f"Epoch {epoch + 1}: Test Acc = {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            torch.save(student, 'distilled_student_full.pth')
            print(f"New best student model saved with acc {best_acc:.2f}%")

    print(f"Knowledge distillation finished. Best test accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    main()