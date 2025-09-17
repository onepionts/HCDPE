# -*- coding: utf-8 -*-
"""
Created on Sun May 18 19:13:57 2025

@author: 黄蔚栋
"""

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 解决OpenMP冲突问题

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt


class GradCAM:
    def __init__(self, model, target_layer):
        """
        初始化Grad-CAM

        参数:
            model: 预训练的PyTorch模型
            target_layer: 要计算CAM的目标层(通常是最后一个卷积层)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 注册钩子
        self._register_hooks()

    def _register_hooks(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        def forward_hook(module, input, output):
            self.activations = output.detach()

        # 获取目标层并注册钩子
        target_layer = self._get_target_layer()
        target_layer.register_forward_hook(forward_hook)
        # 使用新的register_full_backward_hook替代旧的register_backward_hook
        target_layer.register_full_backward_hook(backward_hook)

    def _get_target_layer(self):
        # 递归查找目标层
        module = self.model
        for name in self.target_layer.split('.'):
            module = getattr(module, name)
        return module

    def _preprocess_image(self, img_path, img_size=(224, 224)):
        """预处理输入图像"""
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        return img_tensor, img

    def generate_cam(self, img_path, target_class=None):
        """生成类激活图"""
        # 预处理图像
        input_tensor, orig_img = self._preprocess_image(img_path)

        # 前向传播
        output = self.model(input_tensor)

        # 如果没有指定目标类，则使用预测类
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        # 反向传播
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # 计算权重
        gradients = self.gradients.squeeze()
        activations = self.activations.squeeze()
        weights = torch.mean(gradients, dim=(1, 2))

        # 计算CAM
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        # 应用ReLU并归一化
        cam = torch.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam = cam.numpy()

        # 调整大小以匹配原始图像
        cam = cv2.resize(cam, orig_img.size)

        return cam

    def visualize(self, img_path, target_class=None, save_path=None):
        """可视化Grad-CAM结果"""
        # 生成CAM
        cam = self.generate_cam(img_path, target_class)

        # 读取原始图像
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))

        # 创建热力图
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # 叠加图像
        superimposed_img = heatmap * 0.4 + img * 0.6
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

        # 显示结果
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(superimposed_img)
        plt.title('Grad-CAM')
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)

        plt.show()


# 示例用法
if __name__ == "__main__":
    # 加载预训练模型(使用新的weights参数替代pretrained)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()

    # 创建Grad-CAM实例
    grad_cam = GradCAM(model, target_layer='layer4')

    # 指定图像路径和可选的类别索引
    img_path = 'D:\\1\\1.jpg'  # 替换为你的图像路径
    grad_cam.visualize(img_path, save_path='D:\\1\\2.jpg')