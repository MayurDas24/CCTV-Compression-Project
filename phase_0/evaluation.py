import argparse
import csv
import json
import logging
import pathlib
from typing import Dict, Any

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Evaluation")


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def compute_ssim_fast(img1: np.ndarray, img2: np.ndarray) -> float:
    """Fast SSIM estimation using downsampled grayscale maps."""
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Downsample frame size by 50% to accelerate spatial filter convolutions
    g1 = cv2.resize(g1, (0, 0), fx=0.5, fy=0.5).astype(np.float64)
    g2 = cv2.resize(g2, (0, 0), fx=0.5, fy=0.5).astype(np.float64)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = cv2.GaussianBlur(g1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(g2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(g1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(g2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(g1 * g2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.mean(ssim_map))


class MetricEvaluator:
    def __init__(self, original_path: pathlib.Path):
        self.original_path = original_path

    def evaluate(self, compressed_path: pathlib.Path, method_label: str) -> Dict[str, Any]:
        cap_orig = cv2.VideoCapture(str(self.original_path))
        cap_comp = cv2.VideoCapture(str(compressed_path))

        psnr_vals = []
        ssim_vals = []

        frame_idx = 0
        evaluated_count = 0

        while cap_orig.isOpened() and cap_comp.isOpened():
            ret1, frame1 = cap_orig.read()
            ret2, frame2 = cap_comp.read()

            if not ret1 or not ret2:
                break
            
            # Evaluate every 5th frame for high performance
            if frame_idx % 5 == 0:
                if frame1.shape != frame2.shape:
                    frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

                psnr_vals.append(compute_psnr(frame1, frame2))
                ssim_vals.append(compute_ssim_fast(frame1, frame2))
                evaluated_count += 1

            frame_idx += 1

        cap_orig.release()
        cap_comp.release()

        orig_size = self.original_path.stat().st_size
        comp_size = compressed_path.stat().st_size
        reduction = ((orig_size - comp_size) / orig_size) * 100.0 if orig_size > 0 else 0.0

        return {
            "method": method_label,
            "original_size_mb": round(orig_size / (1024 * 1024), 3),
            "compressed_size_mb": round(comp_size / (1024 * 1024), 3),
            "storage_reduction_percent": round(reduction, 2),
            "mean_psnr": round(float(np.mean(psnr_vals)), 2) if psnr_vals else 0.0,
            "mean_ssim": round(float(np.mean(ssim_vals)), 4) if ssim_vals else 0.0,
            "frames_evaluated": evaluated_count
        }


def main():
    parser = argparse.ArgumentParser(description="Phase 0 Metric Evaluator")
    parser.add_argument("--original", required=True, type=pathlib.Path, help="Original pristine video file")
    parser.add_argument("--baseline", required=True, type=pathlib.Path, help="Baseline standard H.265 encoded video")
    parser.add_argument("--adaptive", required=True, type=pathlib.Path, help="Adaptive ROI encoded video")
    parser.add_argument("--output-csv", default="results/results.csv", type=pathlib.Path, help="Output metrics CSV file")
    args = parser.parse_args()

    evaluator = MetricEvaluator(args.original)
    
    logger.info("Evaluating Baseline standard video...")
    baseline_res = evaluator.evaluate(args.baseline, "H265_Baseline")
    
    logger.info("Evaluating Adaptive compression video...")
    adaptive_res = evaluator.evaluate(args.adaptive, "H265_Adaptive")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = list(baseline_res.keys())
    file_exists = args.output_csv.exists()

    with open(args.output_csv, mode="w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(baseline_res)
        writer.writerow(adaptive_res)

    logger.info(f"Evaluation finished. Output recorded to {args.output_csv}")
    print(json.dumps([baseline_res, adaptive_res], indent=2))


if __name__ == "__main__":
    main()