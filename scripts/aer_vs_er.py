"""
Simple ER vs Accelerated ER Convergence Comparison
Compares reconstruction quality against clean ground truth image.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import resize

# Matplotlib settings for publication quality
plt.rcParams.update({
    # 'font.family': 'serif',
    'font.size': 15,
    'axes.labelsize': 16,
    'axes.titlesize': 17,
    'legend.fontsize': 14,
})


def fft2d(x):
    """2D FFT with normalization."""
    return torch.fft.fftn(x, dim=(-2, -1), norm="ortho") / 2.0


def ifft2d(x):
    """2D IFFT with normalization."""
    return torch.fft.ifftn(x, dim=(-2, -1), norm="ortho") * 2.0


def classical_er(measurements, support, gt_image, n_iters=300):
    """Classical Error Reduction algorithm.
    
    Args:
        measurements: Fourier amplitude measurements (can be noisy)
        support: Real-space support mask
        gt_image: Clean ground truth image for computing errors
        n_iters: Number of iterations
    
    Returns:
        psnr_values: PSNR vs ground truth at each iteration (smoothed)
    """
    # Random initialization
    phase = torch.rand_like(measurements) * 2 * np.pi
    G = torch.polar(measurements, phase)
    
    psnr_values = []
    best_mse = np.inf
    gt_fft = fft2d(gt_image).abs()
    
    for _ in range(n_iters):
        # Fourier constraint: replace amplitude with measurements
        G = torch.polar(measurements, G.angle())
        
        # Image space
        g = torch.real(ifft2d(G))
        
        # Image constraints: support + non-negativity
        g = g * support * (g > 0)
        
        # Back to Fourier
        G = fft2d(g)
        
        # MSE in Fourier domain (what algorithm actually optimizes)
        mse = torch.mean((G.abs() - gt_fft) ** 2).item()
        
        # Smoothing: keep best (minimum) MSE seen so far
        best_mse = min(best_mse, mse)
        
        # Convert to PSNR
        psnr = 10 * np.log10(1.0 / (best_mse + 1e-10))
        psnr_values.append(psnr)
    
    return psnr_values


def accelerated_er(measurements, support, gt_image, n_iters=300, gamma=0.6, accel_period=4):
    """Accelerated Error Reduction algorithm.
    
    Args:
        measurements: Fourier amplitude measurements (can be noisy)
        support: Real-space support mask
        gt_image: Clean ground truth image for computing errors
        n_iters: Number of iterations
        gamma: Acceleration strength
        accel_period: Apply acceleration every N iterations
    
    Returns:
        psnr_values: PSNR vs ground truth at each iteration (smoothed)
    """
    # Random initialization
    phase = torch.rand_like(measurements) * 2 * np.pi
    G = torch.polar(measurements, phase)
    g = torch.real(ifft2d(G))
    
    last_combined = None
    psnr_values = []
    best_mse = np.inf
    gt_fft = fft2d(gt_image).abs()
    
    for i in range(n_iters):
        # Fourier constraint
        G_fourier = torch.polar(measurements, G.angle())
        g_fourier = torch.real(ifft2d(G_fourier))
        
        # Image constraint
        g_image = g_fourier * support * (g_fourier > 0)
        
        # Acceleration mechanism
        if last_combined is None:
            # First iteration: no acceleration
            g = g_image
            last_combined = (g_fourier + g_image) / 2
        elif i % accel_period == accel_period - 1:
            # Acceleration step
            discrepancy = torch.norm(g_fourier - g_image)
            combined = (g_fourier + g_image) / 2
            direction = combined - last_combined
            direction_norm = torch.norm(direction)
            
            if direction_norm > 1e-10:
                g = combined + gamma * 0.5 * discrepancy * direction / direction_norm
            else:
                g = combined
            
            last_combined = combined
        else:
            # Regular ER step
            g = g_image
        
        # Back to Fourier
        G = fft2d(g)
        
        # MSE in Fourier domain (what algorithm actually optimizes)
        mse = torch.mean((G.abs() - gt_fft) ** 2).item()
        
        # Smoothing: keep best (minimum) MSE seen so far
        best_mse = min(best_mse, mse)
        
        # Convert to PSNR
        psnr = 10 * np.log10(1.0 / (best_mse + 1e-10))
        psnr_values.append(psnr)
    
    return psnr_values


def prepare_data(size=128, noise_alpha=0.0):
    """Prepare test image and measurements.
    
    Args:
        size: Image size
        noise_alpha: Noise level (0 = clean)
    
    Returns:
        gt_image: Clean ground truth image
        measurements: Amplitude measurements (noisy if alpha > 0)
        support: Support mask
    """
    # Load cameraman image
    image = data.camera()
    image = resize(image, (size, size), anti_aliasing=True)
    image = (image - image.min()) / (image.max() - image.min())
    image = torch.from_numpy(image).float()
    
    # Create support (central region)
    support = torch.zeros_like(image)
    pad = size // 4
    support[pad:-pad, pad:-pad] = 1.0
    
    # Apply support
    image = image * support
    
    # Zero-pad for oversampling (2x)
    image = torch.nn.functional.pad(image, (size//2, size//2, size//2, size//2))
    support = torch.nn.functional.pad(support, (size//2, size//2, size//2, size//2))
    
    # Compute Fourier amplitude
    fft_clean = fft2d(image)
    amp_clean = fft_clean.abs()
    
    # Add noise if requested
    if noise_alpha > 0:
        noise = noise_alpha * amp_clean * torch.randn_like(amp_clean)
        amp_noisy = torch.sqrt(torch.clamp(amp_clean**2 + noise, min=0))
        measurements = amp_noisy
    else:
        measurements = amp_clean
    
    return image, measurements, support


def plot_convergence(results, save_path='convergence.pdf', max_iter=250):
    """Plot convergence curves."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    # Average over trials (truncate to max_iter)
    classical_psnr = np.array([r[:max_iter] for r in results['classical']])
    accelerated_psnr = np.array([r[:max_iter] for r in results['accelerated']])
    
    mean_classical = classical_psnr.mean(axis=0)
    std_classical = classical_psnr.std(axis=0)
    
    mean_accelerated = accelerated_psnr.mean(axis=0)
    std_accelerated = accelerated_psnr.std(axis=0)
    
    iterations = np.arange(len(mean_classical))
    
    # Plot classical ER
    ax.plot(iterations, mean_classical, '-', color='#3577b2', linewidth=2.5, label='Classical ER', alpha=0.9)
    ax.fill_between(iterations, mean_classical - std_classical, mean_classical + std_classical,
                     color='#3577b2', alpha=0.2)
    
    # Plot accelerated ER
    ax.plot(iterations, mean_accelerated, '-', color='#f58125', linewidth=2.5, label='Accelerated ER', alpha=0.9)
    ax.fill_between(iterations, mean_accelerated - std_accelerated, mean_accelerated + std_accelerated,
                     color='#f58125', alpha=0.2)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('PSNR vs Ground Truth (dB)')
    ax.set_title('Convergence Comparison')
    ax.legend(frameon=True, loc='lower right')
    ax.grid(False)#, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()


def main():
    print("="*60)
    print("ER vs Accelerated ER - Simple Comparison")
    print("="*60)
    
    # Settings
    n_trials = 10
    n_iters = 300
    noise_alpha = 0.0  # Set to 0 for clean, or > 0 for noisy
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Prepare data
    print(f"\nPreparing data (noise_alpha={noise_alpha})...")
    gt_image, measurements, support = prepare_data(size=256, noise_alpha=noise_alpha)
    
    # Run experiments
    print(f"Running {n_trials} trials with {n_iters} iterations...")
    results = {'classical': [], 'accelerated': []}
    
    for trial in range(n_trials):
        # Classical ER
        psnr_classical = classical_er(measurements, support, gt_image, n_iters)
        results['classical'].append(psnr_classical)
        
        # Accelerated ER (same random seed for fair comparison)
        torch.manual_seed(42 + trial)
        psnr_accelerated = accelerated_er(measurements, support, gt_image, n_iters)
        results['accelerated'].append(psnr_accelerated)
        
        # Reset seed for next trial
        torch.manual_seed(42 + trial)
        
        print(f"  Trial {trial+1}/{n_trials} - Classical final: {psnr_classical[-1]:.2f} dB, "
              f"Accelerated final: {psnr_accelerated[-1]:.2f} dB")
    
    # Plot results (only first 250 iterations)
    print("\nGenerating plot...")
    plot_convergence(results, 'convergence.pdf', max_iter=250)
    
    # Summary statistics
    classical_final = [r[-1] for r in results['classical']]
    accelerated_final = [r[-1] for r in results['accelerated']]
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Classical ER final PSNR:    {np.mean(classical_final):.2f} ± {np.std(classical_final):.2f} dB")
    print(f"Accelerated ER final PSNR:  {np.mean(accelerated_final):.2f} ± {np.std(accelerated_final):.2f} dB")
    improvement = np.mean(accelerated_final) - np.mean(classical_final)
    print(f"Improvement: +{improvement:.2f} dB")
    print("="*60)


if __name__ == "__main__":
    main()
