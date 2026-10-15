import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False
import numpy as np

# -------------------------- 完全独立的模型定义，和训练时结构100%对齐 --------------------------
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

# -------------------------- 和训练阶段完全一致的预处理，绝对不会出现推理偏差 --------------------------
transform_base = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -------------------------- 初始化设备和模型 --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CatDogCNN().to(device)
# 加载最优训练权重
model.load_state_dict(torch.load("best_catdog.pth", map_location=device))
model.eval()
print("✅ 模型加载完成，可以开始推理")

# -------------------------- 独立推理主函数 --------------------------
def predict_image(img_path):
    """
    输入本地图片路径，自动完成预处理，输出分类结果、置信度，同时可视化处理后的图片
    """
    # 读取图片并转RGB格式，兼容RGBA、灰度图等异常格式
    img = Image.open(img_path).convert("RGB")
    # 执行和训练完全一致的预处理
    processed = transform_base(img).unsqueeze(0).to(device)
    
    # 关闭梯度计算，加速推理
    with torch.no_grad():
        output = model(processed)
        # 转成概率分布
        prob = torch.softmax(output, dim=1)[0]
        pred_label = torch.argmax(prob).item()
    
    # 反归一化还原图片用于可视化
    vis_img = processed.squeeze(0).cpu() * torch.tensor([0.229,0.224,0.225]).reshape(3,1,1) + torch.tensor([0.485,0.456,0.406]).reshape(3,1,1)
    vis_img = vis_img.permute(1,2,0).numpy().clip(0,1)
    
    # 输出结果
    result = "猫" if pred_label == 0 else "狗"
    confidence = prob[pred_label].item()
    print(f"📊 推理结果：{result}，置信度：{confidence:.2%}")
    
    # 展示结果图
    plt.figure(figsize=(6,6))
    plt.imshow(vis_img)
    plt.title(f"预测结果: {result}\n置信度: {confidence:.2%}", fontsize=14)
    plt.axis("off")
    plt.show()
    
    return result, confidence

# -------------------------- 在这里替换成你自己的测试图片路径直接运行 --------------------------
if __name__ == "__main__":
    # 把这里的路径改成你本地任意一张猫狗图片的路径即可
    predict_image("C:/Users/Lenovo/Desktop/Jotang-ml-task3/image2.jpg")
