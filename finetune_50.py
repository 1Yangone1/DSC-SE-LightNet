import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import DSCSELightNet

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
    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
        last_layer = model.classifier[-1]
        if isinstance(last_layer, nn.Linear) and last_layer.out_features != num_classes:
            print(f"Fixing classifier: from {last_layer.out_features} to {num_classes}")
            in_features = last_layer.in_features
            new_layer = nn.Linear(in_features, num_classes)
            device = next(model.parameters()).device
            new_layer = new_layer.to(device)
            model.classifier[-1] = new_layer
    return model

def main():
    BATCH_SIZE = 64
    EPOCHS = 20
    LEARNING_RATE = 0.0001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", DEVICE)

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

    # 加载剪枝50%的模型
    model = torch.load('real_pruned_model_50.pth', weights_only=False)
    model = model.to(DEVICE)
    model = fix_classifier_head(model, 10)

    # 评估剪枝后未微调准确率（可选）
    # 直接开始微调
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(trainloader, desc=f"Fine-tune Epoch {epoch+1}/{EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix(loss=running_loss/(total/BATCH_SIZE), acc=100.*correct/total)

        acc = evaluate(model, testloader, DEVICE)
        print(f"Epoch {epoch+1}: Test Acc = {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'finetuned_pruned_model_50.pth')
            print(f"New best model saved with acc {best_acc:.2f}%")

    print(f"Fine-tuning finished. Best test accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()