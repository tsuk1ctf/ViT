"""
eval_fr.py — 有参考(Full-Reference, FR)图像质量评估

包含以下三个有参考指标(需要 ground-truth 图像):
    - PSNR  (Peak Signal-to-Noise Ratio)         越高越好
    - SSIM  (Structural Similarity Index)        越高越好(范围 [-1, 1])
    - LPIPS (Learned Perceptual Image Patch Similarity)  越低越好

默认在评估前把 pred 和 GT 都 resize 到 512×512,保证不同方法的输出
在同一尺寸基准下可比;传入 --size 0 可关闭。
"""
import os
import cv2
import csv
import numpy as np
from natsort import natsorted
import lpips
import torch
from skimage.metrics import structural_similarity as ssim
from argparse import ArgumentParser


class ImageEvaluator:
    def __init__(self, use_lpips=False,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 unified_size=512):
        """
        有参考图像质量评估器
        :param use_lpips: 是否启用 LPIPS 评估
        :param device: 计算设备
        :param unified_size: 评估前统一 resize 到此尺寸(默认 512),保证
                             不同方法的输出在同一基准下可比;传入 None
                             则保持原始 GT 尺寸
        """
        self.device = device
        self.unified_size = unified_size  # 统一评估尺寸
        self.lpips_model = lpips.LPIPS(net='alex').to(device) if use_lpips else None

    def _check_image_shapes(self, img1, img2):
        if img1.shape != img2.shape:
            raise ValueError(f"Image shapes mismatch: {img1.shape} vs {img2.shape}")

    def read_image(self, path, convert_rgb=True):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if convert_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def psnr(self, img1, img2, max_val=255.0):
        self._check_image_shapes(img1, img2)
        mse = np.mean((img1 - img2) ** 2)
        return 10 * np.log10((max_val ** 2) / (mse + 1e-10))

    def ssim(self, img1, img2, win_size=11, multichannel=True):
        self._check_image_shapes(img1, img2)
        return ssim(img1, img2, win_size=win_size,
                    multichannel=multichannel, channel_axis=-1,
                    data_range=255)

    def lpips(self, img1, img2):
        if self.lpips_model is None:
            raise RuntimeError("LPIPS model not initialized. Use --use_lpips flag")
        self._check_image_shapes(img1, img2)
        img1_tensor = lpips.im2tensor(img1).to(self.device)
        img2_tensor = lpips.im2tensor(img2).to(self.device)
        with torch.no_grad():
            distance = self.lpips_model(img1_tensor, img2_tensor)
        return distance.item()

    def evaluate_pair(self, pred_path, gt_path):
        pred = self.read_image(pred_path)
        gt = self.read_image(gt_path)

        # 统一到 512×512:不同分辨率下 PSNR/SSIM/LPIPS 不可直接比较
        # pred 和 GT 都 resize 到该尺寸,保证在公平条件下计算
        if self.unified_size is not None:
            target_hw = (self.unified_size, self.unified_size)
            pred = cv2.resize(pred, target_hw, interpolation=cv2.INTER_CUBIC)
            gt = cv2.resize(gt, target_hw, interpolation=cv2.INTER_CUBIC)
        else:
            # 未启用统一尺寸时，仍把 pred 对齐到 GT 的尺寸（兼容旧行为）
            if pred.shape != gt.shape:
                h, w = gt.shape[:2]
                pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_CUBIC)

        results = {
            'psnr': self.psnr(pred, gt),
            'ssim': self.ssim(pred, gt),
            'lpips': self.lpips(pred, gt) if self.lpips_model else None
        }
        return results

    def evaluate_folder(self, pred_dir, gt_dir, extensions=('png', 'jpg', 'jpeg'), output_csv=None):
        results = []
        pairs = self._pair_files(pred_dir, gt_dir, extensions)

        for pred_path, gt_path in pairs:
            try:
                res = self.evaluate_pair(pred_path, gt_path)
                res['filename'] = os.path.basename(pred_path)
                results.append(res)
                print(f"Processed: {res['filename']} | PSNR: {res['psnr']:.2f} | SSIM: {res['ssim']:.4f}" +
                      (f" | LPIPS: {res['lpips']:.4f}" if res['lpips'] is not None else ""))
            except Exception as e:
                print(f"Error processing {pred_path}: {str(e)}")

        aggregated = self._aggregate_results(results)

        # 新增CSV输出功能
        if output_csv:
            self._save_results_to_csv(results, aggregated, output_csv)

        return aggregated

        # return self._aggregate_results(results)

    def _save_results_to_csv(self, results, aggregated, output_path):
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # 写入表头
            headers = ['Filename', 'PSNR', 'SSIM']
            if self.lpips_model:
                headers.append('LPIPS')
            writer.writerow(headers)

            # 写入每条结果
            for res in results:
                row = [
                    res['filename'],
                    f"{res['psnr']:.2f}",
                    f"{res['ssim']:.4f}"
                ]
                if self.lpips_model:
                    row.append(f"{res['lpips']:.4f}")
                writer.writerow(row)

            # 写入平均结果
            avg_row = [
                'Average',
                f"{aggregated['psnr']:.2f}" if aggregated['psnr'] is not None else "N/A",
                f"{aggregated['ssim']:.4f}" if aggregated['ssim'] is not None else "N/A"
            ]
            if self.lpips_model:
                avg_row.append(f"{aggregated['lpips']:.4f}" if aggregated['lpips'] is not None else "N/A")
            writer.writerow(avg_row)

        print(f"\nResults saved to {output_path}")

    def _pair_files(self, pred_dir, gt_dir, extensions=('png', 'jpg', 'jpeg')):
        pairs = []
        extensions = tuple([ext.lower() for ext in extensions])  # 统一转小写

        # 构建GT文件名映射（不含扩展名 -> 完整路径）
        gt_map = {}
        for f in os.listdir(gt_dir):
            if f.lower().endswith(extensions):
                gt_base = os.path.splitext(f)[0].lower()
                gt_map[gt_base] = os.path.join(gt_dir, f)

        pred_files = natsorted([
            f for f in os.listdir(pred_dir)
            if f.lower().endswith(extensions)
        ])

        for fname in pred_files:
            base_name = os.path.splitext(fname)[0].lower()
            # 精确匹配文件名（不含扩展名）
            if base_name in gt_map:
                pred_path = os.path.join(pred_dir, fname)
                gt_path = gt_map[base_name]
                pairs.append((pred_path, gt_path))
                print(f"Matched: {pred_path} -> {gt_path}")
            else:
                print(f"Warning: No GT found for {fname}")
        return pairs
    def _aggregate_results(self, results):
        valid_results = [r for r in results if None not in r.values()]

        if not valid_results:
            print("\nError: No valid image pairs found!")
            return {
                'psnr': None,
                'ssim': None,
                'lpips': None
            }

        aggregated = {
            'psnr': np.mean([r['psnr'] for r in valid_results]),
            'ssim': np.mean([r['ssim'] for r in valid_results]),
            'lpips': np.mean(
                [r['lpips'] for r in valid_results if r['lpips'] is not None]) if self.lpips_model else None
        }
        print("\nAggregated Results:")
        print(f"PSNR: {aggregated['psnr']:.2f} dB")
        print(f"SSIM: {aggregated['ssim']:.4f}")
        if self.lpips_model:
            print(f"LPIPS: {aggregated['lpips']:.4f}")
        return aggregated

if __name__ == "__main__":
    parser = ArgumentParser(description="有参考图像质量评估 (PSNR/SSIM/LPIPS)")
    parser.add_argument("--pred", type=str, required=True, help="预测结果文件夹路径")
    parser.add_argument("--gt", type=str, required=True, help="真实图像文件夹路径")
    parser.add_argument("--output", type=str, default=None, help="输出CSV文件路径（可选）")
    parser.add_argument("--use_lpips", action="store_true", help="是否启用LPIPS计算")
    # 默认统一到 512×512,保证与其他方法在同一基准下比较
    parser.add_argument("--size", type=int, default=512,
                        help="统一评估尺寸(默认 512×512),传入 0 表示保持 GT 尺寸")
    args = parser.parse_args()

    # 检查路径有效性
    if not os.path.exists(args.pred):
        raise FileNotFoundError(f"预测目录不存在: {args.pred}")
    if not os.path.exists(args.gt):
        raise FileNotFoundError(f"真实图像目录不存在: {args.gt}")

    unified_size = args.size if args.size > 0 else None
    evaluator = ImageEvaluator(use_lpips=args.use_lpips, unified_size=unified_size)
    print(f"统一评估尺寸: {unified_size if unified_size else 'GT 原始尺寸'}")
    final_results = evaluator.evaluate_folder(args.pred, args.gt, output_csv=args.output)
