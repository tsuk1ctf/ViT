"""
eval_nr.py — 无参考(No-Reference, NR)图像质量评估

包含以下三个无参考指标(均越低越好):
    - NIQE    (Natural Image Quality Evaluator)
    - BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)
    - PIQE    (Perception-based Image Quality Evaluator)

默认在评估前把所有图像统一 resize 到 512×512,保证不同方法的输出
在同一尺寸基准下可比;传入 --size 0 可关闭。
"""
import os
import csv
import numpy as np
import torch
import pyiqa
from natsort import natsorted
from PIL import Image
from termcolor import colored


class RobustQualityEvaluator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu',
                 unified_size=512):
        """
        无参考图像质量评估器
        :param device: 计算设备
        :param unified_size: 评估前统一 resize 到此尺寸(默认 512),保证
                             不同方法的输出在同一基准下可比;传入 None
                             则保持原始尺寸
        """
        self.device = device
        self.unified_size = unified_size  # 统一评估尺寸
        self.metrics = {
            'niqe': pyiqa.create_metric('niqe', device=device),
            'brisque': pyiqa.create_metric('brisque', device=device),
            'piqe': pyiqa.create_metric('piqe', device=device)
        }

    def _validate_image(self, img_path):
        """强化图像验证逻辑,支持JPEG/PNG/BMP格式检测"""
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"文件不存在: {img_path}")
        if not os.path.isfile(img_path):
            raise ValueError(f"不是有效文件: {img_path}")
        if not os.access(img_path, os.R_OK):
            raise PermissionError(f"无读取权限: {img_path}")

        try:
            # 读取文件头验证格式
            with open(img_path, 'rb') as f:
                header = f.read(8)
                png_signature = b'\x89PNG\r\n\x1a\n'
                jpeg_signature = b'\xFF\xD8'
                bmp_signature = b'BM'  # BMP文件签名

                if header.startswith(png_signature):
                    file_type = 'PNG'
                elif header[:2] == jpeg_signature:
                    file_type = 'JPEG'
                elif header[:2] == bmp_signature:
                    file_type = 'BMP'
                else:
                    raise ValueError("不支持的文件格式,仅支持JPEG/PNG/BMP")
            # 图像内容验证
            with Image.open(img_path) as img:
                img.load()
                # 统一转换为 RGB 模式
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # 尺寸验证（移除固定尺寸限制，支持各种尺寸）
                min_size = 32  # PIQE最小尺寸要求
                if min(img.size) < min_size:
                    raise ValueError(f"图像尺寸过小，最小需要{min_size}x{min_size}，实际尺寸: {img.size}")

                # 统一到 512×512:NIQE/BRISQUE/PIQE 对图像分辨率敏感,
                # 不同尺寸下的得分不可直接比较
                if self.unified_size is not None and img.size != (self.unified_size, self.unified_size):
                    img = img.resize((self.unified_size, self.unified_size), Image.BICUBIC)

                # 像素值验证
                arr = np.array(img)
                if arr.min() < 0 or arr.max() > 255:
                    raise ValueError(f"像素值越界 (范围: {arr.min()}-{arr.max()})")

                return img
        except Exception as e:
            raise RuntimeError(f"图像验证失败 ({type(e).__name__}): {str(e)}")

    def _safe_metric_calculation(self, metric_name, img_tensor):
        """带异常处理的指标计算"""
        try:
            # PIQE的特殊尺寸要求
            if metric_name == 'piqe':
                h, w = img_tensor.shape[2], img_tensor.shape[3]
                if h < 32 or w < 32:
                    raise ValueError(f"图像尺寸过小 ({h}x{w})")

            result = self.metrics[metric_name](img_tensor).item()

            # 结果合理性检查
            if metric_name == 'piqe' and not (0 <= result <= 100):
                raise ValueError(f"PIQE值异常: {result}")

            return result
        except Exception as e:
            print(colored(f"[{metric_name.upper()}警告] {str(e)}", "yellow"))
            return 100.0

    def evaluate_image(self, img_path):
        """评估单个图像"""
        try:
            img = self._validate_image(img_path)
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            img_tensor = img_tensor.to(self.device)

            return {
                'niqe': self._safe_metric_calculation('niqe', img_tensor),
                'brisque': self._safe_metric_calculation('brisque', img_tensor),
                'piqe': self._safe_metric_calculation('piqe', img_tensor)
            }
        except FileNotFoundError as e:
            print(colored(f"[文件缺失] {os.path.basename(img_path)}", "red"))
        except PermissionError as e:
            print(colored(f"[权限不足] {os.path.basename(img_path)}", "red"))
        except ValueError as e:
            if "不支持的文件格式" in str(e):
                print(colored(f"[格式错误] {os.path.basename(img_path)}: {str(e)}", "yellow"))
            else:
                print(colored(f"[验证失败] {os.path.basename(img_path)}: {str(e)}", "yellow"))
        except Exception as e:
            print(colored(f"[未知错误] {os.path.basename(img_path)}: {str(e)}", "magenta"))

        return self._error_result()

    def _error_result(self):
        """错误时返回默认值"""
        return {'niqe': 100.0, 'brisque': 100.0, 'piqe': 100.0}

    def _collect_images(self, img_dir):
        """递归收集目录中的所有图像文件"""
        img_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        all_images = []
        for root, _, files in os.walk(img_dir):
            for f in files:
                if os.path.splitext(f)[1].lower() in img_extensions:
                    all_images.append(os.path.join(root, f))
        return natsorted(all_images)

    def batch_evaluate(self, img_dir, output_csv):
        """批量评估目录中的图像（支持递归子目录）"""
        results = []
        valid_count = 0
        error_count = 0

        print(colored(f"开始评估目录: {img_dir}", "blue"))

        all_images = self._collect_images(img_dir)
        print(colored(f"共找到 {len(all_images)} 张图像", "blue"))

        for img_path in all_images:
            # 使用相对路径作为文件名标识
            f = os.path.relpath(img_path, img_dir)

            result = self.evaluate_image(img_path)
            result['filename'] = f
            results.append(result)

            if all(v == 100.0 for v in result.values()):
                error_count += 1
                status = colored("失败", "red")
            else:
                valid_count += 1
                status = colored("成功", "green")

            print(f"{status} | {f.ljust(20)} | "
                  f"NIQE: {result['niqe']:6.2f} | "
                  f"BRISQUE: {result['brisque']:6.2f} | "
                  f"PIQE: {result['piqe']:6.2f}")

        # 保存结果
        self._save_csv(results, output_csv)
        print(colored(f"\n处理完成: 有效 {valid_count} / 错误 {error_count}", "cyan"))
        return self._aggregate(results)

    def _aggregate(self, results):
        """计算平均值，过滤无效结果"""
        valid_results = [r for r in results if not all(v == 100.0 for v in r.values())]
        if not valid_results:
            return {'niqe': 100.0, 'brisque': 100.0, 'piqe': 100.0}

        return {
            'niqe': np.mean([r['niqe'] for r in valid_results]),
            'brisque': np.mean([r['brisque'] for r in valid_results]),
            'piqe': np.mean([r['piqe'] for r in valid_results])
        }

    def _save_csv(self, results, path):
        """保存CSV结果文件"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:  # 支持中文
            writer = csv.writer(f)
            writer.writerow(['文件名', 'NIQE', 'BRISQUE', 'PIQE', '状态'])

            for r in results:
                status = "有效" if not all(v == 100.0 for v in r.values()) else "错误"
                writer.writerow([
                    r['filename'],
                    f"{r['niqe']:.2f}",
                    f"{r['brisque']:.2f}",
                    f"{r['piqe']:.2f}",
                    status
                ])

            # 添加平均行
            avg = self._aggregate(results)
            writer.writerow([
                '平均值',
                f"{avg['niqe']:.2f}",
                f"{avg['brisque']:.2f}",
                f"{avg['piqe']:.2f}",
                ""
            ])


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="无参考图像质量评估 (NIQE/BRISQUE/PIQE)")
    parser.add_argument("--img", type=str, required=True, help="增强图像文件夹路径")
    parser.add_argument("--output", type=str, default=None, help="输出CSV文件路径（可选）")
    # 默认统一到 512×512,保证与其他方法在同一基准下比较
    parser.add_argument("--size", type=int, default=512,
                        help="统一评估尺寸(默认 512×512),传入 0 表示不 resize")
    args = parser.parse_args()

    unified_size = args.size if args.size > 0 else None
    evaluator = RobustQualityEvaluator(unified_size=unified_size)

    # 自动检测设备
    print(colored(f"使用计算设备: {evaluator.device}", "blue"))
    print(colored(f"统一评估尺寸: {unified_size if unified_size else '原始尺寸'}", "blue"))

    results = evaluator.batch_evaluate(
        img_dir=args.img,
        output_csv=args.output if args.output else f"./results_niqe.csv"
    )

    print(colored("\n质量评估结果:", "magenta", attrs=['bold']))
    print(f"NIQE 平均值: {results['niqe']:.2f}")
    print(f"BRISQUE 平均值: {results['brisque']:.2f}")
    print(f"PIQE 平均值: {results['piqe']:.2f}")