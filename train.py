import torch
import torch.optim as optim
from config import config
from dataset import get_dataloader
from model import DarkEnhancer
from utils import LossCalculator
from torch.utils.tensorboard import SummaryWriter
import os

os.environ['ALBUMENTATIONS_CHECK_VERSION'] = '0'  # 禁用albumentations版本检查


def train():
    # 初始化训练环境
    device = torch.device(config.device)
    model = DarkEnhancer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    loss_calculator = LossCalculator(device)

    # 准备数据加载器
    train_loader = get_dataloader(config.data_root, 'train', config.batch_size)
    val_loader = get_dataloader(config.data_root, 'val', config.batch_size)

    # 初始化TensorBoard
    writer = SummaryWriter(config.tensorboard_dir)

    # 训练循环
    for epoch in range(config.epochs):
        model.train()
        total_train_loss = 0.0

        # 训练批次迭代
        for idx, (low_imgs, high_imgs) in enumerate(train_loader):
            low_imgs = low_imgs.to(device)
            high_imgs = high_imgs.to(device)

            # 前向传播
            enhanced, mask = model(low_imgs)

            # 计算复合损失
            loss = loss_calculator.total_loss(enhanced, high_imgs, mask)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录损失
            total_train_loss += loss.item()
            if idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{config.epochs}], Step [{idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
                writer.add_scalar('training_loss', loss.item(), epoch * len(train_loader) + idx)

        # 计算平均训练损失
        avg_train_loss = total_train_loss / len(train_loader)

        # 验证阶段
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for low_imgs, high_imgs in val_loader:
                low_imgs = low_imgs.to(device)
                high_imgs = high_imgs.to(device)
                enhanced, mask = model(low_imgs)
                total_val_loss += loss_calculator.total_loss(enhanced, high_imgs, mask).item()

        avg_val_loss = total_val_loss / len(val_loader)

        # 记录指标
        writer.add_scalar('average_train_loss', avg_train_loss, epoch)
        writer.add_scalar('average_val_loss', avg_val_loss, epoch)

        # 可视化示例结果（每5轮）
        if epoch % 5 == 0:
            # 获取训练批次样本
            batch = next(iter(train_loader))
            low_imgs, high_imgs = batch
            low_imgs = low_imgs.to(device)
            high_imgs = high_imgs.to(device)

            # 生成增强图像
            enhanced, _ = model(low_imgs)

            # 记录图像到TensorBoard
            writer.add_image('low_image', low_imgs[0], epoch, dataformats='CHW')
            writer.add_image('enhanced_image', enhanced[0], epoch, dataformats='CHW')
            writer.add_image('target_image', high_imgs[0], epoch, dataformats='CHW')

        # 保存模型检查点
        torch.save(model.state_dict(), config.checkpoint_path)
        print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")