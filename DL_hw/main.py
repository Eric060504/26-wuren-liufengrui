import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import os

# 数据增强和预处理
transform_train = transforms.Compose([
    transforms.Resize((64, 64)),  # 将图像调整为64x64
    transforms.RandomCrop(64, padding=8),  # 随机裁剪
    transforms.RandomHorizontalFlip(),     # 随机水平翻转
    transforms.RandomRotation(15),         # 随机旋转±15度
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet标准化参数
])

transform_test = transforms.Compose([
    transforms.Resize((64, 64)),  # 将图像调整为64x64
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
data_dir = 'd:\homework\car team\hw\DL\dataset'  # 数据集根目录

# 训练集
train_dataset = datasets.ImageFolder(
    root=os.path.join(data_dir, 'train'),
    transform=transform_train
)

# 测试集1
test1_dataset = datasets.ImageFolder(
    root=os.path.join(data_dir, 'test1'),
    transform=transform_test
)

# 测试集2
test2_dataset = datasets.ImageFolder(
    root=os.path.join(data_dir, 'test2'),
    transform=transform_test
)

# 数据加载器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test1_loader = DataLoader(test1_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test2_loader = DataLoader(test2_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 类别信息
classes = train_dataset.classes  # 应该是 ['blue', 'red', 'yellow'] 或按文件夹顺序的类别名
num_classes = len(classes)
print(f"类别: {classes}")

# 残差块定义
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # 第一个卷积层: 3x3卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 第二个卷积层: 3x3卷积
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut连接，如果输入输出通道数不同则使用1x1卷积调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x  # 保存输入用于残差连接
        
        # 第一个卷积操作
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # 第二个卷积操作
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 残差连接
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out

# 基于ResNet的网络模型
class ResNet(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet, self).__init__()
        # 输入: 3x64x64 (RGB图像，尺寸64x64)
        self.in_channels = 16
        
        # 初始卷积层: 3x3卷积，输出通道16
        # 输入: 3x64x64，输出: 16x64x64
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        
        # 第一个残差块组 (2个残差块)
        # 输入: 16x64x64，输出: 16x64x64
        self.layer1 = self._make_layer(ResidualBlock, 16, 2, stride=1)
        
        # 第二个残差块组 (2个残差块)
        # 输入: 16x64x64，输出: 32x32x32 (通过stride=2进行下采样)
        self.layer2 = self._make_layer(ResidualBlock, 32, 2, stride=2)
        
        # 第三个残差块组 (2个残差块)
        # 输入: 32x32x32，输出: 64x16x16 (通过stride=2进行下采样)
        self.layer3 = self._make_layer(ResidualBlock, 64, 2, stride=2)
        
        # 全局平均池化: 64x16x16 -> 64x1x1
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层1: 64 -> 32
        self.fc1 = nn.Linear(64, 32)
        # 全连接层2: 32 -> num_classes (3个类别)
        self.fc2 = nn.Linear(32, num_classes)
        
        
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 初始卷积层
        out = self.conv1(x)    # 输出: 16x64x64
        out = self.bn1(out)
        out = self.relu(out)
        
        # 残差块组
        out = self.layer1(out)  # 输出: 16x64x64
        out = self.layer2(out)  # 输出: 32x32x32
        out = self.layer3(out)  # 输出: 64x16x16
        
        # 全局平均池化
        out = self.avg_pool(out)  # 输出: 64x1x1
        out = out.view(out.size(0), -1)  # 展平: 64
        
        # 全连接层
        out = self.fc1(out)     # 输出: 32
        out = self.relu(out)
        out = self.fc2(out)     # 输出: num_classes (3)
        
        return out

# 初始化模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# 训练函数
def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()  # 清零梯度
        
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, targets)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)  # 获取预测结果
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 打印训练进度
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx*len(inputs)}/{len(train_loader.dataset)} '
                  f'({100.*batch_idx/len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    train_acc = 100. * correct / total
    train_loss_avg = train_loss / len(train_loader)
    print(f'Train Epoch: {epoch} \tAverage Loss: {train_loss_avg:.6f}, Accuracy: {correct}/{total} ({train_acc:.2f}%)')
    return train_loss_avg, train_acc

# 验证函数
def test(loader, dataset_name="Test"):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():  # 关闭梯度计算
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 收集预测结果和真实标签用于后续分析
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # 计算整体准确率
    overall_acc = 100. * correct / total
    test_loss_avg = test_loss / len(loader)
    print(f'\n{dataset_name} set: Average loss: {test_loss_avg:.4f}, '
          f'Overall Accuracy: {correct}/{total} ({overall_acc:.2f}%)')
    
    # 计算每类的准确率
    print(f'{dataset_name} set per-class accuracy:')
    report = classification_report(all_targets, all_preds, target_names=classes, output_dict=True)
    for cls in classes:
        print(f'Class {cls}: {report[cls]["precision"]*100:.2f}%')
    
    return test_loss_avg, overall_acc, report

# 主函数
def main():
    epochs = 30  # 训练轮数
    best_acc = 0  # 记录最佳准确率
    
    # 记录训练过程
    train_losses = []
    train_accs = []
    test1_losses = []
    test1_accs = []
    
    for epoch in range(1, epochs+1):
        train_loss, train_acc = train(epoch)
        test1_loss, test1_acc, _ = test(test1_loader, "Test1")
        
        # 学习率调整
        scheduler.step(test1_loss)
        
        # 保存最好的模型
        if test1_acc > best_acc:
            best_acc = test1_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"模型已更新并保存，当前最佳准确率: {best_acc:.2f}%")
        
        # 记录训练指标
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test1_losses.append(test1_loss)
        test1_accs.append(test1_acc)
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test1_losses, label='Test1 Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test1_accs, label='Test1 Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()
    
    # 使用最佳模型在两个测试集上进行最终验证
    print("\n--- Final evaluation with best model ---")
    model.load_state_dict(torch.load('best_model.pth'))
    test(test1_loader, "Test1")
    test(test2_loader, "Test2")

if __name__ == '__main__':
    main()
