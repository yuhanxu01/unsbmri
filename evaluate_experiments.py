#!/usr/bin/env python
"""
Comprehensive Evaluation Script for Noise-Adaptive Experiments

This script evaluates and compares all experiments by:
1. Computing quantitative metrics (PSNR, SSIM, noise level)
2. Generating comparison visualizations
3. Creating summary tables and plots
4. Statistical analysis of improvements

Usage:
    python evaluate_experiments.py --results_dir ./results
    python evaluate_experiments.py --results_dir ./results --save_path ./evaluation_report.html
"""

import argparse
import os
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import pandas as pd
from tqdm import tqdm
import h5py
import json

# Import noise estimation
import sys
sys.path.append('.')
from noise_estimation import estimate_noise_mad


class ExperimentEvaluator:
    """Evaluates and compares multiple experiments"""

    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.experiments = self._discover_experiments()
        self.metrics = {}

    def _discover_experiments(self):
        """Discover all experiment directories"""
        experiments = []
        for exp_dir in self.results_dir.iterdir():
            if exp_dir.is_dir():
                experiments.append(exp_dir.name)
        experiments.sort()
        print(f"Found {len(experiments)} experiments: {experiments}")
        return experiments

    def load_images(self, exp_name):
        """Load generated images from an experiment"""
        exp_dir = self.results_dir / exp_name
        images = {}

        # Look for image files (adjust pattern based on your test.py output)
        for img_path in exp_dir.glob("*.png"):
            # Assuming naming: {sample_id}_fake.png
            sample_id = img_path.stem.replace('_fake', '').replace('_real', '')

            if 'fake' in img_path.stem:
                if sample_id not in images:
                    images[sample_id] = {}
                images[sample_id]['generated'] = plt.imread(img_path)
            elif 'real' in img_path.stem and 'A' in img_path.stem:
                if sample_id not in images:
                    images[sample_id] = {}
                images[sample_id]['input'] = plt.imread(img_path)
            elif 'real' in img_path.stem and 'B' in img_path.stem:
                if sample_id not in images:
                    images[sample_id] = {}
                images[sample_id]['target'] = plt.imread(img_path)

        print(f"  Loaded {len(images)} image sets for {exp_name}")
        return images

    def compute_metrics(self, exp_name):
        """Compute metrics for an experiment"""
        images = self.load_images(exp_name)
        metrics = {
            'psnr': [],
            'ssim': [],
            'input_noise': [],
            'output_noise': [],
            'noise_reduction_ratio': []
        }

        for sample_id, img_dict in tqdm(images.items(), desc=f"Evaluating {exp_name}"):
            if 'generated' not in img_dict or 'target' not in img_dict:
                continue

            generated = img_dict['generated']
            target = img_dict['target']
            input_img = img_dict.get('input', None)

            # Convert to grayscale if needed
            if generated.ndim == 3 and generated.shape[2] > 1:
                generated = np.mean(generated, axis=2)
            if target.ndim == 3 and target.shape[2] > 1:
                target = np.mean(target, axis=2)
            if input_img is not None and input_img.ndim == 3 and input_img.shape[2] > 1:
                input_img = np.mean(input_img, axis=2)

            # Ensure same size
            min_h = min(generated.shape[0], target.shape[0])
            min_w = min(generated.shape[1], target.shape[1])
            generated = generated[:min_h, :min_w]
            target = target[:min_h, :min_w]

            # Normalize to [0, 1]
            generated = (generated - generated.min()) / (generated.max() - generated.min() + 1e-8)
            target = (target - target.min()) / (target.max() - target.min() + 1e-8)

            # Compute PSNR and SSIM
            psnr_val = psnr(target, generated, data_range=1.0)
            ssim_val = ssim(target, generated, data_range=1.0)

            metrics['psnr'].append(psnr_val)
            metrics['ssim'].append(ssim_val)

            # Compute noise levels
            output_noise = estimate_noise_mad(generated)
            metrics['output_noise'].append(output_noise)

            if input_img is not None:
                if input_img.ndim == 3 and input_img.shape[2] > 1:
                    input_img = np.mean(input_img, axis=2)
                input_noise = estimate_noise_mad(input_img)
                metrics['input_noise'].append(input_noise)
                if input_noise > 0:
                    metrics['noise_reduction_ratio'].append(output_noise / input_noise)

        # Aggregate statistics
        results = {}
        for key, values in metrics.items():
            if len(values) > 0:
                results[f'{key}_mean'] = np.mean(values)
                results[f'{key}_std'] = np.std(values)
                results[f'{key}_median'] = np.median(values)
            else:
                results[f'{key}_mean'] = None
                results[f'{key}_std'] = None
                results[f'{key}_median'] = None

        return results

    def evaluate_all(self):
        """Evaluate all experiments"""
        print("\nEvaluating all experiments...")
        print("=" * 80)

        for exp_name in self.experiments:
            print(f"\n{exp_name}:")
            metrics = self.compute_metrics(exp_name)
            self.metrics[exp_name] = metrics

            # Print summary
            if metrics['psnr_mean'] is not None:
                print(f"  PSNR: {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f} dB")
            if metrics['ssim_mean'] is not None:
                print(f"  SSIM: {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}")
            if metrics['output_noise_mean'] is not None:
                print(f"  Output Noise: {metrics['output_noise_mean']:.4f} ± {metrics['output_noise_std']:.4f}")
            if metrics['noise_reduction_ratio_mean'] is not None:
                noise_reduction_pct = (1 - metrics['noise_reduction_ratio_mean']) * 100
                print(f"  Noise Reduction: {noise_reduction_pct:.1f}%")

        return self.metrics

    def create_comparison_plots(self, save_dir='./evaluation_plots'):
        """Create comparison plots"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

        # Prepare data
        exp_names = []
        psnr_means = []
        psnr_stds = []
        ssim_means = []
        ssim_stds = []
        noise_reduction = []

        for exp_name, metrics in self.metrics.items():
            exp_names.append(exp_name)
            psnr_means.append(metrics.get('psnr_mean', 0))
            psnr_stds.append(metrics.get('psnr_std', 0))
            ssim_means.append(metrics.get('ssim_mean', 0))
            ssim_stds.append(metrics.get('ssim_std', 0))

            ratio = metrics.get('noise_reduction_ratio_mean', 1.0)
            noise_reduction.append((1 - ratio) * 100 if ratio else 0)

        # Plot 1: PSNR Comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(exp_names))
        ax.bar(x, psnr_means, yerr=psnr_stds, capsize=5, alpha=0.7, color='steelblue')
        ax.set_xlabel('Experiment', fontsize=12)
        ax.set_ylabel('PSNR (dB)', fontsize=12)
        ax.set_title('PSNR Comparison Across Experiments', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(exp_names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'psnr_comparison.png', dpi=150)
        plt.close()

        # Plot 2: SSIM Comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x, ssim_means, yerr=ssim_stds, capsize=5, alpha=0.7, color='coral')
        ax.set_xlabel('Experiment', fontsize=12)
        ax.set_ylabel('SSIM', fontsize=12)
        ax.set_title('SSIM Comparison Across Experiments', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(exp_names, rotation=45, ha='right')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'ssim_comparison.png', dpi=150)
        plt.close()

        # Plot 3: Noise Reduction
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['red' if nr < 0 else 'green' for nr in noise_reduction]
        ax.bar(x, noise_reduction, alpha=0.7, color=colors)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Experiment', fontsize=12)
        ax.set_ylabel('Noise Reduction (%)', fontsize=12)
        ax.set_title('Noise Reduction Across Experiments', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(exp_names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'noise_reduction_comparison.png', dpi=150)
        plt.close()

        # Plot 4: Combined metrics radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        # Normalize metrics to [0, 1]
        psnr_norm = np.array(psnr_means) / max(psnr_means)
        ssim_norm = np.array(ssim_means)
        noise_red_norm = np.array(noise_reduction) / 100

        angles = np.linspace(0, 2 * np.pi, 3, endpoint=False).tolist()
        angles += angles[:1]

        for i, exp_name in enumerate(exp_names):
            values = [psnr_norm[i], ssim_norm[i], noise_red_norm[i]]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=exp_name)
            ax.fill(angles, values, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(['PSNR\n(normalized)', 'SSIM', 'Noise\nReduction'], fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title('Multi-Metric Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(save_dir / 'radar_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nPlots saved to {save_dir}/")

    def generate_report(self, save_path='./evaluation_report.html'):
        """Generate HTML report"""
        df = pd.DataFrame(self.metrics).T

        # Create HTML
        html = f"""
        <html>
        <head>
            <title>Noise-Adaptive Experiments Evaluation Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                h1 {{
                    color: #333;
                    text-align: center;
                }}
                h2 {{
                    color: #555;
                    border-bottom: 2px solid #ddd;
                    padding-bottom: 10px;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    background-color: white;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                th {{
                    background-color: #4CAF50;
                    color: white;
                }}
                tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                .metric {{
                    font-weight: bold;
                }}
                .improvement {{
                    color: green;
                }}
                .degradation {{
                    color: red;
                }}
                img {{
                    max-width: 100%;
                    margin: 20px 0;
                }}
            </style>
        </head>
        <body>
            <h1>Noise-Adaptive Experiments Evaluation Report</h1>

            <h2>Summary Statistics</h2>
            {df.to_html(float_format='%.4f')}

            <h2>Visualizations</h2>
            <h3>PSNR Comparison</h3>
            <img src="evaluation_plots/psnr_comparison.png" alt="PSNR Comparison">

            <h3>SSIM Comparison</h3>
            <img src="evaluation_plots/ssim_comparison.png" alt="SSIM Comparison">

            <h3>Noise Reduction</h3>
            <img src="evaluation_plots/noise_reduction_comparison.png" alt="Noise Reduction">

            <h3>Multi-Metric Comparison</h3>
            <img src="evaluation_plots/radar_comparison.png" alt="Radar Comparison">

            <h2>Key Findings</h2>
            <ul>
        """

        # Find best experiments
        if 'psnr_mean' in df.columns:
            best_psnr = df['psnr_mean'].idxmax()
            html += f"<li><span class='metric'>Best PSNR:</span> {best_psnr} ({df.loc[best_psnr, 'psnr_mean']:.2f} dB)</li>"

        if 'ssim_mean' in df.columns:
            best_ssim = df['ssim_mean'].idxmax()
            html += f"<li><span class='metric'>Best SSIM:</span> {best_ssim} ({df.loc[best_ssim, 'ssim_mean']:.4f})</li>"

        if 'noise_reduction_ratio_mean' in df.columns:
            best_noise = df['noise_reduction_ratio_mean'].idxmin()
            reduction = (1 - df.loc[best_noise, 'noise_reduction_ratio_mean']) * 100
            html += f"<li><span class='metric'>Best Noise Reduction:</span> {best_noise} ({reduction:.1f}%)</li>"

        html += """
            </ul>
        </body>
        </html>
        """

        with open(save_path, 'w') as f:
            f.write(html)

        print(f"\nReport saved to {save_path}")

        # Also save metrics as JSON
        json_path = save_path.replace('.html', '.json')
        with open(json_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics saved to {json_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate noise-adaptive experiments')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Directory containing experiment results')
    parser.add_argument('--save_path', type=str, default='./evaluation_report.html',
                       help='Path to save evaluation report')
    parser.add_argument('--plot_dir', type=str, default='./evaluation_plots',
                       help='Directory to save plots')

    args = parser.parse_args()

    # Create evaluator
    evaluator = ExperimentEvaluator(args.results_dir)

    # Evaluate all experiments
    evaluator.evaluate_all()

    # Create plots
    evaluator.create_comparison_plots(args.plot_dir)

    # Generate report
    evaluator.generate_report(args.save_path)

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print(f"  - Report: {args.save_path}")
    print(f"  - Plots: {args.plot_dir}/")
    print("=" * 80)


if __name__ == '__main__':
    main()
