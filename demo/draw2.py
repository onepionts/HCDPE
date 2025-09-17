import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 放在所有import之前

# 然后继续其他导入
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def simulate_feature_extraction(image_path, output_size=(200, 200)):
    # 1. 读取图像并预处理
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("无法加载图像，请检查路径是否正确")

    # 转换为RGB并调整大小
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))  # 调整为常见CNN输入尺寸

    # 2. 模拟卷积操作
    # 转换为PyTorch tensor并归一化
    tensor_img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    tensor_img = tensor_img.unsqueeze(0)  # 添加batch维度

    # 创建模拟卷积层
    conv_layers = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 第一层卷积
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 第二层卷积
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 第三层卷积
        nn.ReLU()
    )

    # 3. 应用卷积获取特征图
    with torch.no_grad():
        features = conv_layers(tensor_img)

    # 4. 处理特征图用于可视化
    # 合并所有特征通道（取平均值）
    feature_map = features.squeeze(0).mean(dim=0).numpy()

    # 归一化到0-255范围
    feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min()) * 255
    feature_map = feature_map.astype(np.uint8)

    # 调整大小为输出尺寸并应用马赛克效果
    feature_map = cv2.resize(feature_map, output_size, interpolation=cv2.INTER_NEAREST)

    # 应用颜色映射模拟热图效果
    feature_map = cv2.applyColorMap(feature_map, cv2.COLORMAP_JET)

    # 5. 显示原始图像和特征图
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('原始图像')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(feature_map, cv2.COLOR_BGR2RGB))
    plt.title('模拟特征图')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return feature_map


# 使用示例
if __name__ == "__main__":
    image_path = 'D:\\1\\1.jpg'  # 替换为你的图片路径
    feature_map = simulate_feature_extraction(image_path)

    # 保存特征图
    cv2.imwrite('D:\\1\\2.jpg', feature_map)
    print("特征图已保存为 feature_map.jpg")