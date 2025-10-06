import torch
import cv2
import numpy as np
import os
from config import config
from model import DarkEnhancer
from torchvision import transforms


def process_image(model, input_path, output_path):
    # 读取图像
    img = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)
    if img is None:
        raise ValueError(f"Failed to read image from {input_path}")

    # 保存原始尺寸
    original_height, original_width = img.shape[:2]

    # 预处理
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    input_tensor = transform(img).unsqueeze(0).to(config.device)

    # 推理
    with torch.no_grad():
        enhanced, _ = model(input_tensor)

    # 后处理
    enhanced_np = enhanced.squeeze().cpu().numpy().transpose(1, 2, 0)
    enhanced_np = (enhanced_np * 255).astype(np.uint8)

    # 恢复原始尺寸并保存
    enhanced_resized = cv2.resize(enhanced_np, (original_width, original_height))
    cv2.imwrite(output_path, cv2.cvtColor(enhanced_resized, cv2.COLOR_RGB2BGR))


def enhance_folder(input_folder, output_folder):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 初始化模型
    model = DarkEnhancer().to(config.device)
    model.load_state_dict(torch.load(config.checkpoint_path, map_location=config.device))
    model.eval()

    # 处理所有支持格式的图像
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(valid_exts):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                process_image(model, input_path, output_path)
                print(f"Successfully processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")


if __name__ == "__main__":
    # 设置输入输出文件夹
    input_folder = "./test"  # 替换为你的输入文件夹路径
    # output_folder = "./output"  # 替换为你的输出文件夹路径

    enhance_folder(input_folder, output_folder)