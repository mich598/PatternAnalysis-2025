"""
visualization.py

Evaluate and visualize TimeGAN-generated data using:
  1. Distribution similarity (KL divergence)
  2. Visual similarity (SSIM heatmap comparison)
  3. 5 Representative heatmaps for real vs generated LOB depth snapshots

Reference: 
Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," NeurIPS 2019.
"""

from scipy.stats import entropy
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt


def visualization(ori_data, generated_data, n_examples=5):
    """
    Evaluate and visualize TimeGAN-generated data.
    
    Args:
        - ori_data: np.array (n_samples, seq_len, dim)
        - generated_data: np.array (n_samples, seq_len, dim)
        - n_examples: number of real-vs-generated examples to visualize (default=5)
    """

    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)
    assert ori_data.shape == generated_data.shape, "Original and generated data must have the same shape."

    n_samples, seq_len, dim = ori_data.shape

    # ----------------------------
    # 1. Distribution Similarity (KL Divergence)
    # ----------------------------

    def compute_spread_midret(data):
        best_bid = data[:, :, 0]
        best_ask = data[:, :, 1]
        mid = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        mid_ret = np.diff(mid, axis=1) / mid[:, :-1]  # midprice returns
        return spread[:, :-1].flatten(), mid_ret.flatten()

    ori_spread, ori_midret = compute_spread_midret(ori_data)
    gen_spread, gen_midret = compute_spread_midret(generated_data)

    bins = 100
    ori_spread_hist, _ = np.histogram(ori_spread, bins=bins, density=True)
    gen_spread_hist, _ = np.histogram(gen_spread, bins=bins, density=True)
    ori_midret_hist, _ = np.histogram(ori_midret, bins=bins, density=True)
    gen_midret_hist, _ = np.histogram(gen_midret, bins=bins, density=True)

    eps = 1e-10
    kl_spread = entropy(ori_spread_hist + eps, gen_spread_hist + eps)
    kl_midret = entropy(ori_midret_hist + eps, gen_midret_hist + eps)

    print(f"KL Divergence (Spread): {kl_spread:.4f}")
    print(f"KL Divergence (Midprice Return): {kl_midret:.4f}")
    print("Target: KL ≤ 0.1 for good similarity\n")

    # Plot distribution comparisons
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(ori_spread, bins=100, alpha=0.5, label="Original", density=True, color='red')
    plt.hist(gen_spread, bins=100, alpha=0.5, label="Generated", density=True, color='blue')
    plt.title(f"Spread Distribution (KL={kl_spread:.4f})")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(ori_midret, bins=100, alpha=0.5, label="Original", density=True, color='red')
    plt.hist(gen_midret, bins=100, alpha=0.5, label="Generated", density=True, color='blue')
    plt.title(f"Midprice Return Distribution (KL={kl_midret:.4f})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ----------------------------
    # 2. Visual Similarity (SSIM)
    # ----------------------------

    ori_heatmap = np.mean(ori_data, axis=0)
    gen_heatmap = np.mean(generated_data, axis=0)

    ori_norm = (ori_heatmap - np.min(ori_heatmap)) / (np.max(ori_heatmap) - np.min(ori_heatmap) + eps)
    gen_norm = (gen_heatmap - np.min(gen_heatmap)) / (np.max(gen_heatmap) - np.min(gen_heatmap) + eps)

    ssim_value = ssim(ori_norm, gen_norm, data_range=1.0)
    print(f"SSIM (Heatmap Visual Similarity): {ssim_value:.4f}")
    print("Target: SSIM > 0.6 for good visual similarity\n")

    # Plot mean heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im1 = axes[0].imshow(ori_norm, aspect='auto', cmap='hot')
    axes[0].set_title("Original Mean LOB Depth")
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(gen_norm, aspect='auto', cmap='hot')
    axes[1].set_title("Generated Mean LOB Depth")
    fig.colorbar(im2, ax=axes[1])

    plt.suptitle(f"SSIM = {ssim_value:.4f}")
    plt.tight_layout()
    plt.show()

    # ----------------------------
    # 3. Representative Heatmap Examples
    # ----------------------------

    n_examples = min(n_examples, n_samples)
    sample_idx = np.random.choice(n_samples, n_examples, replace=False)

    fig, axes = plt.subplots(n_examples, 2, figsize=(8, 2 * n_examples))
    if n_examples == 1:
        axes = np.array([axes])  # ensure consistent indexing

    for i, idx in enumerate(sample_idx):
        real_ex = ori_data[idx]
        gen_ex = generated_data[idx]

        # Normalize for visualization
        real_norm = (real_ex - np.min(real_ex)) / (np.max(real_ex) - np.min(real_ex) + eps)
        gen_norm = (gen_ex - np.min(gen_ex)) / (np.max(gen_ex) - np.min(gen_ex) + eps)

        ssim_example = ssim(real_norm, gen_norm, data_range=1.0)

        axes[i, 0].imshow(real_norm, aspect='auto', cmap='viridis')
        axes[i, 0].set_title(f"Real Sample #{idx}")
        axes[i, 1].imshow(gen_norm, aspect='auto', cmap='viridis')
        axes[i, 1].set_title(f"Generated Sample #{idx}\nSSIM={ssim_example:.3f}")

    plt.suptitle(f"{n_examples} Representative LOB Heatmaps (Real vs Generated)")
    plt.tight_layout()
    plt.show()
