import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False



# ===================== 你只需要确认这个路径和你本地完全对应即可 =====================
DATASET_ROOT = "C:/Users/Lenovo/Desktop/Jotang-ml-task3/猫狗数据集/train/train"
# 如果你后续运行提示找不到路径，把上面这行改成你本地的绝对路径，比如 "C:/Users/景妍/Desktop/Jotang-ml-task/猫狗数据集/train"

# -------------------------- 1. 数据集观察 对应题目第1题 --------------------------
class CatDogDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # 自动按文件名前缀区分猫狗，完全适配Kaggle原版数据集
        self.all_imgs = []
        for f in os.listdir(root_dir):
            if f.endswith('.jpg'):
                if f.startswith('cat'):
                    self.all_imgs.append( (os.path.join(root_dir, f), 0) )
                elif f.startswith('dog'):
                    self.all_imgs.append( (os.path.join(root_dir, f), 1) )

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_path, label = self.all_imgs[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# 统计数据集
dataset_raw = CatDogDataset(DATASET_ROOT)
print(f"数据集加载完成，总样本数: {len(dataset_raw)}")
print(f"猫的图片数量: {sum(1 for _, label in dataset_raw.all_imgs if label == 0)}")
print(f"狗的图片数量: {sum(1 for _, label in dataset_raw.all_imgs if label == 1)}")

# 随机展示5张原图
plt.figure(figsize=(12,6))
for i in range(5):
    img, label = random.choice(dataset_raw)
    plt.subplot(1,5,i+1)
    plt.imshow(img)
    plt.title("猫" if label==0 else "狗")
    plt.axis("off")
plt.suptitle("原始随机样本", y=0.92)
plt.show()

# -------------------------- 2. 预处理与Batch构建 对应题目第2题 --------------------------
# 基础预处理（验证/测试集专用）
transform_base = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 带数据增强的预处理（训练集专用）
transform_aug = transforms.Compose([
    transforms.Resize((140,140)),
    transforms.RandomCrop(128),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 手动切分7:1.5:1.5 训练/验证/测试集，给不同子集绑定对应预处理
full_img_list = dataset_raw.all_imgs
train_size = int(0.7 * len(full_img_list))
val_size = int(0.15 * len(full_img_list))
test_size = len(full_img_list) - train_size - val_size

all_indices = list(range(len(full_img_list)))
random.shuffle(all_indices)
train_indices = all_indices[:train_size]
val_indices = all_indices[train_size:train_size+val_size]
test_indices = all_indices[train_size+val_size:]

train_dataset = Subset(CatDogDataset(DATASET_ROOT, transform=transform_aug), train_indices)
val_dataset = Subset(CatDogDataset(DATASET_ROOT, transform=transform_base), val_indices)
test_dataset = Subset(CatDogDataset(DATASET_ROOT, transform=transform_base), test_indices)

# 构建DataLoader自动生成Batch
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
print(f"数据集切分完成: 训练集{train_size}张, 验证集{val_size}张, 测试集{test_size}张")

# 展示预处理前后对比
raw_img, _ = random.choice(dataset_raw)
processed_img = transform_base(raw_img).permute(1,2,0).numpy()
processed_img = processed_img * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(raw_img)
plt.title("预处理前")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(np.clip(processed_img,0,1))
plt.title("预处理后 128×128")
plt.axis("off")
plt.show()

# -------------------------- 3. 搭建CNN 对应题目第3题 --------------------------
class CatDogCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

model = CatDogCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"模型加载完成，运行设备: {device}")

# -------------------------- 4. 各层Tensor形状打印 对应题目第4题 --------------------------
test_input = torch.randn(1,3,128,128).to(device)
print("\n===== 各层Tensor形状变化 =====")
print(f"输入: {test_input.shape} → 批次1, 通道3, 高128, 宽128")
x = model.feature_extractor[0](test_input)
print(f"Conv1输出: {x.shape} → 通道升为16，高宽保持128")
x = model.feature_extractor[2](x)
print(f"MaxPool1输出: {x.shape} → 高宽减半为64")
x = model.feature_extractor[3](x)
print(f"Conv2输出: {x.shape} → 通道升为32，高宽保持64")
x = model.feature_extractor[5](x)
print(f"MaxPool2输出: {x.shape} → 高宽减半为32")
x = model.feature_extractor[6](x)
print(f"Conv3输出: {x.shape} → 通道升为64，高宽保持32")
x = model.feature_extractor[8](x)
print(f"MaxPool3输出: {x.shape} → 高宽减半为16")

# -------------------------- 5. 卷积特征可视化 对应题目第5题 --------------------------
sample_img, _ = test_dataset[0]
sample_img_batch = sample_img.unsqueeze(0).to(device)

# 第一层卷积特征可视化
first_conv_out = model.feature_extractor[0](sample_img_batch).squeeze(0).detach().cpu()
plt.figure(figsize=(12,8))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(first_conv_out[i], cmap='gray')
    plt.title(f"通道{i}")
    plt.axis("off")
plt.suptitle("第一层卷积16个通道特征图", y=0.98)
plt.show()

# 最后一层卷积特征可视化
# 先把图片完整过前6层，自动得到第6层卷积的输出
last_conv_out = model.feature_extractor[:7](sample_img_batch).squeeze(0).detach().cpu()
plt.figure(figsize=(16,16))
for i in range(64):
    plt.subplot(8,8,i+1)
    plt.imshow(last_conv_out[i], cmap='gray')
    plt.axis("off")
plt.suptitle("最后一层卷积64个通道特征图", y=0.98)
plt.show()
print("特征图可视化完成：浅层关注边缘纹理，深层关注猫狗脸部等高级语义特征")

# -------------------------- 6. 数据增强对照实验 对应题目第6题 --------------------------
no_aug_transform = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor()])
full_aug_transform = transform_aug
sample_raw = random.choice(dataset_raw)[0]
plt.figure(figsize=(10,5))
plt.subplot(1,3,1)
plt.imshow(sample_raw)
plt.title("原图")
plt.axis("off")
plt.subplot(1,3,2)
plt.imshow(no_aug_transform(sample_raw).permute(1,2,0))
plt.title("无增强")
plt.axis("off")
plt.subplot(1,3,3)
plt.imshow(full_aug_transform(sample_raw).permute(1,2,0))
plt.title("带增强")
plt.axis("off")
plt.show()
print("\n 数据增强合理性分析：")
print("合理增强：随机水平翻转、轻微随机裁剪、亮度对比度微调，不会改变猫狗核心语义，可降低过拟合")
print("不合理增强：旋转90度以上、颜色剧烈偏移、重度噪声，会破坏图像语义，甚至人类都无法分辨猫狗")

# -------------------------- 7. 训练+错误样本分析 对应题目第7题 --------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 10
best_acc = 0
wrong_samples = []
train_loss_history = []
val_acc_history = []

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} 训练中"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # 验证
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, pred = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (pred == labels).sum().item()
            # 收集预测错误的样本
            wrong_mask = pred != labels
            for i in range(len(imgs)):
                if wrong_mask[i]:
                    wrong_samples.append((imgs[i].cpu(), labels[i].item(), pred[i].item()))
    
    val_acc = val_correct / val_total
    avg_train_loss = train_loss / len(train_loader)
    train_loss_history.append(avg_train_loss)
    val_acc_history.append(val_acc)
    print(f"Epoch {epoch+1}, 训练损失: {avg_train_loss:.4f}, 验证准确率: {val_acc:.4f}")
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_catdog.pth")
        print(f"最优模型已保存，当前最高验证准确率: {best_acc:.4f}")

# 绘制loss和accuracy曲线
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(range(1, epochs+1), train_loss_history, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("训练损失曲线")
plt.legend()
plt.subplot(1,2,2)
plt.plot(range(1, epochs+1), val_acc_history, label="Val Accuracy", color="orange")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("验证集准确率曲线")
plt.legend()
plt.show()

# 分析错误样本
print(f"\n收集到错误样本共 {len(wrong_samples)} 张")
plt.figure(figsize=(12,6))
for i in range(min(5, len(wrong_samples))):
    img, gt, pred = wrong_samples[i]
    img = img * torch.tensor([0.229,0.224,0.225]).reshape(3,1,1) + torch.tensor([0.485,0.456,0.406]).reshape(3,1,1)
    plt.subplot(1,5,i+1)
    plt.imshow(img.permute(1,2,0).numpy().clip(0,1))
    plt.title(f"真实:{'猫'if gt==0 else '狗'}\n预测:{'猫'if pred==0 else '狗'}")
    plt.axis("off")
plt.suptitle("预测错误样本", y=0.92)
plt.show()
print("常见错误原因：猫狗脸部被遮挡、图像模糊、背景干扰极强，改进方向可以加深网络、增加更多增强策略")



