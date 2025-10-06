import torch


class Config:
    # 数据相关配置
    data_root = "./data/LOL_dataset"  # 数据集根目录路径
    input_size = 224  # 模型输入尺寸
    batch_size = 8  # 训练批次大小

    # 训练相关参数
    epochs = 100  # 训练总轮数
    lr = 1e-4  # 学习率
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择设备

    # 模型保存路径
    checkpoint_path = "model.pth"  # 模型保存文件名
    tensorboard_dir = "./runs"  # TensorBoard日志目录


config = Config()  # 创建配置实例供其他模块导入