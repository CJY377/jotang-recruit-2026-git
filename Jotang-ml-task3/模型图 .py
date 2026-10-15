import torch
import torch.nn as nn
from torchinfo import summary

# 这里直接把你自己在主文件里写的模型类原封不动复制过来
class CatDogClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # 这里替换成你自己的卷积、池化、全连接层结构
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 1)
        )
    def forward(self, x):
        return self.layers(x)

if __name__ == "__main__":
    # 自动选设备，不用从别的文件导入
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 直接初始化模型
    model = CatDogClassifier().to(device)

    # 生成模型结构
    summary(
        model=model,
        input_size=(1, 3, 128, 128),
        device=device,
        col_names=["input_size", "output_size", "num_params", "kernel_size"],
        col_width=20,
        row_settings=["var_names"]
    )
    print("模型结构图生成完成，没有任何导入报错")
