from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示成方块的问题
plt.rcParams['font.family'] = 'sans-serif'

# 1. 读取图片
img = Image.open(r"C:\Users\Lenovo\Desktop\Jotang-ml-task3\image.png").convert("RGB")

# 2. 转NumPy数组
img_np = np.array(img)
print("===== NumPy 数组信息 =====")
print("shape:", img_np.shape)
print("dtype:", img_np.dtype)
print("最大值:", img_np.max())
print("最小值:", img_np.min())
# 打印坐标(100,100)位置像素的RGB值
print("坐标(100,100)像素的RGB数值:", img_np[100, 100, :])

# 3. 转PyTorch Tensor
img_tensor = torch.from_numpy(img_np)
print("\n===== PyTorch Tensor 信息 =====")
print("shape:", img_tensor.shape)
print("dtype:", img_tensor.dtype)
print("最大值:", img_tensor.max().item())
print("最小值:", img_tensor.min().item())

# 打印Tensor下对应像素的RGB值
print("坐标(100,100)像素的RGB数值:", img_tensor[100, 100, :])



def my_conv2d(image, kernel, stride=1, padding=0):
    """
    手写二维卷积函数，支持灰度图和RGB图
    :param image: 输入图像，HWC格式的NumPy数组，支持单通道灰度图和3通道RGB图
    :param kernel: 卷积核，二维矩阵
    :param stride: 滑动步长，默认1
    :param padding: 边缘补零的像素数，默认0
    :return: 卷积输出的特征图
    """
    # 先获取基础参数
    H, W = image.shape[0], image.shape[1]
    # 兼容灰度图（单通道）和RGB图（三通道）
    if len(image.shape) == 2:
        C = 1
        image = image.reshape(H, W, 1)
    else:
        C = image.shape[2]
    
    k_h, k_w = kernel.shape[0], kernel.shape[1]

    # 对图像做边缘补零
    if padding > 0:
        pad_img = np.zeros((H + 2*padding, W + 2*padding, C), dtype=image.dtype)
        pad_img[padding:padding+H, padding:padding+W, :] = image
    else:
        pad_img = image
    
    new_H = (pad_img.shape[0] - k_h) // stride + 1
    new_W = (pad_img.shape[1] - k_w) // stride + 1

    # 初始化输出特征图
    output = np.zeros((new_H, new_W, C), dtype=np.float32)

    # 核心卷积滑动计算过程
    for c in range(C):
        for i in range(new_H):
            for j in range(new_W):
                # 取出当前卷积核覆盖的图像局部区域
                region = pad_img[i*stride : i*stride + k_h, j*stride : j*stride + k_w, c]
                # 对应元素相乘求和
                output[i, j, c] = np.sum(region * kernel)
    
    # 如果是单通道灰度图，去掉多余的通道维度
    if C == 1:
        output = output.squeeze()
    return output


# 测试示例：用3x3边缘检测卷积核测试
if __name__ == "__main__":
    from PIL import Image
    # 读取测试图转灰度图
    test_img = np.array(Image.open(r"C:\Users\Lenovo\Desktop\Jotang-ml-task3\image.png").convert("L"))
    # 定义边缘检测卷积核
    edge_kernel = np.array([[-1, -1, -1],
                            [-1,  8, -1],
                            [-1, -1, -1]])
    result = my_conv2d(test_img, edge_kernel, stride=1, padding=1)
    print("卷积输出的shape：", result.shape)




# 先把你之前写好的my_conv2d函数放在这里

# ---------------------- 生成4种卷积核 ----------------------
# 1. 3×3均值模糊核：所有位置数值都是1，最后整体除以9保证亮度不变
avg_kernel = np.ones((3,3), dtype=np.float32) / 9

# 2. 5×5高斯模糊核：手动生成符合二维高斯分布的核
def get_gaussian_kernel(k_size=5, sigma=1.5):
    kernel = np.zeros((k_size, k_size), dtype=np.float32)
    center = k_size // 2
    for i in range(k_size):
        for j in range(k_size):
            # 二维高斯公式计算权重
            kernel[i,j] = np.exp(-((i-center)**2 + (j-center)**2)/(2*sigma**2))
    # 归一化，保证所有数值加起来等于1，图片整体亮度不会变
    kernel = kernel / np.sum(kernel)
    return kernel
gaussian_kernel = get_gaussian_kernel(5, 1.5)

# 3. 锐化卷积核
sharpen_kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
], dtype=np.float32)

# 4. Sobel边缘检测核
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)
sobel_y = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
], dtype=np.float32)

# ---------------------- 读取原图并做4种卷积处理 ----------------------
# 读取RGB原图
img = np.array(Image.open(r"C:\Users\Lenovo\Desktop\Jotang-ml-task3\image.png").convert("RGB"))

# 通用后处理函数：把卷积后超出0-255范围的数值裁剪，转成合法图片格式
def post_process(arr):
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr

# 均值模糊处理
avg_result = post_process(my_conv2d(img, avg_kernel, stride=1, padding=1))
Image.fromarray(avg_result).save("avg_blur.png")

# 高斯模糊处理
gaussian_result = post_process(my_conv2d(img, gaussian_kernel, stride=1, padding=2))
Image.fromarray(gaussian_result).save("gaussian_blur.png")

# 锐化处理
sharpen_result = post_process(my_conv2d(img, sharpen_kernel, stride=1, padding=1))
Image.fromarray(sharpen_result).save("sharpen.png")

# Sobel边缘检测：先转灰度图再分别计算两个方向，最后合并
gray_img = np.array(Image.open(r"C:\Users\Lenovo\Desktop\Jotang-ml-task3\image.png").convert("L"))
sobel_x_result = my_conv2d(gray_img, sobel_x, stride=1, padding=1)
sobel_y_result = my_conv2d(gray_img, sobel_y, stride=1, padding=1)
# 合并两个方向的边缘，取绝对值避免正负抵消
sobel_total = np.sqrt(sobel_x_result**2 + sobel_y_result**2)
sobel_result = post_process(sobel_total)
Image.fromarray(sobel_result).save("sobel_edge.png")

# 把原图和4张结果图拼在一起对比展示
import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
plt.subplot(231),plt.imshow(img),plt.title("原图")
plt.subplot(232),plt.imshow(avg_result),plt.title("均值模糊")
plt.subplot(233),plt.imshow(gaussian_result),plt.title("高斯模糊")
plt.subplot(234),plt.imshow(sharpen_result),plt.title("锐化")
plt.subplot(235),plt.imshow(sobel_result, cmap="gray"),plt.title("Sobel边缘检测")
plt.show()

# 测试3×3卷积核，stride=1的情况
print("===== Valid模式（padding=0） =====")
valid_result = my_conv2d(gray_img, avg_kernel, stride=1, padding=0)
print(f"原图尺寸：{gray_img.shape}")
print(f"Valid模式输出尺寸：{valid_result.shape}")
print("边缘像素变化：原图最外圈的1像素边缘直接被丢弃，输出图比原图小2个像素，边缘没有补零的伪影")

print("\n===== Same模式（padding=1） =====")
same_result = my_conv2d(gray_img, avg_kernel, stride=1, padding=1)
print(f"Same模式输出尺寸：{same_result.shape}")
print("边缘像素变化：原图最外圈的像素也能被卷积核完整计算，输出图和原图尺寸完全一致，边缘补的零会让最外围一圈像素亮度略暗")



