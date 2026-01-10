"""
GPU-accelerated scene detection algorithms using CuPy.

This module provides GPU-accelerated versions of the detection algorithms
for identifying hard cuts in video content. It uses CuPy for CUDA-based
parallel processing to achieve significant speedups over the CPU implementation.

All functions are designed to produce results identical to their CPU counterparts
within a tolerance of 1e-5 for floating point comparisons.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Lazy-loaded CuPy module
_cp = None


def _get_cupy():
    """Lazily import and cache CuPy module."""
    global _cp
    if _cp is None:
        try:
            import cupy as cp

            _cp = cp
        except ImportError as e:
            raise ImportError(
                "CuPy is required for GPU acceleration.\n"
                "Install with: uv pip install cupy-cuda13x\n"
                "Or with pip: pip install cupy-cuda13x"
            ) from e
    return _cp


# =============================================================================
# Array Type Detection Helper
# =============================================================================


def _is_cupy_array(arr) -> bool:
    """
    Check if an array is a CuPy array (already on GPU).

    This function checks the type without importing CuPy if not already loaded,
    which allows graceful handling of mixed array types.

    Args:
        arr: Array to check (NumPy, CuPy, or other).

    Returns:
        bool: True if arr is a CuPy ndarray, False otherwise.
    """
    # Check if CuPy is loaded and if array is CuPy type
    if _cp is not None:
        return isinstance(arr, _cp.ndarray)
    # If CuPy not loaded, check by module name
    return type(arr).__module__.startswith("cupy")


def _stack_frames_to_gpu(frames: list):
    """
    Stack frames and ensure they're on GPU memory.

    Handles both NumPy arrays (transfers to GPU) and CuPy arrays
    (stacks directly on GPU without CPU round-trip).

    Args:
        frames: List of frames as NumPy or CuPy arrays.

    Returns:
        CuPy array of stacked frames with shape (N, H, W, 3).
    """
    cp = _get_cupy()

    if not frames:
        return cp.array([])

    # Check if frames are already on GPU (CuPy arrays)
    if _is_cupy_array(frames[0]):
        # Frames are already on GPU - stack directly
        return cp.stack(frames, axis=0)
    else:
        # Frames are NumPy arrays - transfer to GPU
        frames_stack = np.stack(frames, axis=0)
        return cp.asarray(frames_stack)


# =============================================================================
# Color Conversion Helper Functions
# =============================================================================


def bgr_to_gray_gpu(frames_gpu):
    """
    Convert BGR frames to grayscale on GPU.

    Uses the standard luminosity formula: 0.114*B + 0.587*G + 0.299*R
    This matches OpenCV's cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).

    Args:
        frames_gpu: CuPy array of shape (N, H, W, 3) or (H, W, 3) in BGR format.

    Returns:
        CuPy array of shape (N, H, W) or (H, W) with grayscale values.
    """
    cp = _get_cupy()
    # BGR to grayscale weights (OpenCV compatible)
    weights = cp.array([0.114, 0.587, 0.299], dtype=cp.float32)

    # Handle both single frame and batch
    if frames_gpu.ndim == 3:
        # Single frame: (H, W, 3) -> (H, W)
        gray = cp.tensordot(frames_gpu.astype(cp.float32), weights, axes=([-1], [0]))
    else:
        # Batch: (N, H, W, 3) -> (N, H, W)
        gray = cp.tensordot(frames_gpu.astype(cp.float32), weights, axes=([-1], [0]))

    return gray


def bgr_to_hsv_gpu(frames_gpu):
    """
    Convert BGR frames to HSV color space on GPU.

    This implementation matches OpenCV's cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).

    HSV ranges:
    - H (Hue): 0-179 (OpenCV uses 0-180 scale for uint8)
    - S (Saturation): 0-255
    - V (Value): 0-255

    Args:
        frames_gpu: CuPy array of shape (N, H, W, 3) or (H, W, 3) in BGR format.

    Returns:
        CuPy array of shape (N, H, W, 3) or (H, W, 3) in HSV format.
    """
    cp = _get_cupy()

    # Ensure float for calculations
    frames_float = frames_gpu.astype(cp.float32)

    # Extract B, G, R channels
    if frames_float.ndim == 3:
        b, g, r = frames_float[:, :, 0], frames_float[:, :, 1], frames_float[:, :, 2]
    else:
        b, g, r = frames_float[..., 0], frames_float[..., 1], frames_float[..., 2]

    # Calculate V (max of BGR)
    v = cp.maximum(cp.maximum(r, g), b)

    # Calculate S
    min_rgb = cp.minimum(cp.minimum(r, g), b)
    diff = v - min_rgb

    # Avoid division by zero
    s = cp.where(v > 0, (diff / v) * 255, 0)

    # Calculate H
    h = cp.zeros_like(v)

    # When max == min, H = 0
    # When max == R: H = 60 * (G - B) / diff
    # When max == G: H = 60 * (2 + (B - R) / diff)
    # When max == B: H = 60 * (4 + (R - G) / diff)

    mask_diff = diff > 0

    # Red is max
    mask_r = mask_diff & (v == r)
    h = cp.where(mask_r, 60 * (g - b) / cp.where(diff > 0, diff, 1), h)

    # Green is max
    mask_g = mask_diff & (v == g)
    h = cp.where(mask_g, 60 * (2 + (b - r) / cp.where(diff > 0, diff, 1)), h)

    # Blue is max
    mask_b = mask_diff & (v == b)
    h = cp.where(mask_b, 60 * (4 + (r - g) / cp.where(diff > 0, diff, 1)), h)

    # Normalize H to 0-179 range (OpenCV convention)
    h = cp.where(h < 0, h + 360, h)
    h = h / 2  # Scale from 0-360 to 0-180

    # Stack HSV channels (same operation for both single and batch)
    hsv = cp.stack([h, s, v], axis=-1)

    return hsv.astype(cp.float32)


# =============================================================================
# Histogram Computation Helper Functions
# =============================================================================


def compute_histogram_gpu(channel_gpu, bins: int, range_min: float, range_max: float):
    """
    Compute histogram for a single channel on GPU.

    Args:
        channel_gpu: CuPy array of shape (H, W) containing channel values.
        bins: Number of histogram bins.
        range_min: Minimum value of the histogram range.
        range_max: Maximum value of the histogram range.

    Returns:
        CuPy array of shape (bins,) containing the normalized histogram.
    """
    cp = _get_cupy()

    # Compute histogram using CuPy
    hist, _ = cp.histogram(channel_gpu.flatten(), bins=bins, range=(range_min, range_max))

    # Normalize to 0-1 range
    hist = hist.astype(cp.float32)
    hist_max = cp.max(hist)
    if hist_max > 0:
        hist = hist / hist_max

    return hist


def compute_histogram_batch_gpu(channels_gpu, bins: int, range_min: float, range_max: float):
    """
    Compute histograms for a batch of channels on GPU.

    Uses CuPy's optimized histogram kernel for each frame. While there's a loop
    over frames, each histogram computation is GPU-accelerated, and the batch
    correlation that follows is fully vectorized.

    Args:
        channels_gpu: CuPy array of shape (N, H, W) containing channel values.
        bins: Number of histogram bins.
        range_min: Minimum value of the histogram range.
        range_max: Maximum value of the histogram range.

    Returns:
        CuPy array of shape (N, bins) containing normalized histograms.
    """
    cp = _get_cupy()

    n_frames = channels_gpu.shape[0]
    histograms = cp.zeros((n_frames, bins), dtype=cp.float32)

    # Compute histograms using CuPy's optimized histogram kernel
    # Note: Histogram counting is inherently a scatter operation that requires
    # atomic operations, making full batch parallelization complex without
    # custom CUDA kernels. Each cp.histogram call is still GPU-accelerated.
    for i in range(n_frames):
        hist, _ = cp.histogram(channels_gpu[i].ravel(), bins=bins, range=(range_min, range_max))
        histograms[i] = hist.astype(cp.float32)

    # Normalize all histograms at once (fully vectorized)
    hist_max = cp.max(histograms, axis=1, keepdims=True)
    histograms = cp.where(hist_max > 0, histograms / hist_max, histograms)

    return histograms


def histogram_correlation_gpu(hist1, hist2):
    """
    Compute correlation between two histograms on GPU.

    This matches OpenCV's cv2.compareHist with cv2.HISTCMP_CORREL method.

    Args:
        hist1: CuPy array of shape (bins,) - first histogram.
        hist2: CuPy array of shape (bins,) - second histogram.

    Returns:
        float: Correlation coefficient in range [-1, 1].
    """
    cp = _get_cupy()

    # Subtract means
    mean1 = cp.mean(hist1)
    mean2 = cp.mean(hist2)

    hist1_centered = hist1 - mean1
    hist2_centered = hist2 - mean2

    # Compute correlation
    numerator = cp.sum(hist1_centered * hist2_centered)
    denominator = cp.sqrt(cp.sum(hist1_centered**2) * cp.sum(hist2_centered**2))

    # Avoid division by zero using vectorized where
    correlation = numerator / denominator if denominator > 1e-10 else cp.array(0.0)

    return float(correlation.get()) if hasattr(correlation, "get") else float(correlation)


def histogram_correlation_batch_gpu(hists1, hists2):
    """
    Compute correlations for multiple histogram pairs on GPU without Python loops.

    This vectorized implementation processes all pairs simultaneously, avoiding
    the GPU→CPU synchronization overhead of processing pairs individually.

    Args:
        hists1: CuPy array of shape (N, bins) - first set of histograms.
        hists2: CuPy array of shape (N, bins) - second set of histograms.

    Returns:
        CuPy array of shape (N,) with correlation coefficients in range [-1, 1].
    """
    cp = _get_cupy()

    # Compute means for all histograms at once: (N, bins) -> (N, 1)
    mean1 = cp.mean(hists1, axis=1, keepdims=True)
    mean2 = cp.mean(hists2, axis=1, keepdims=True)

    # Center all histograms
    centered1 = hists1 - mean1
    centered2 = hists2 - mean2

    # Compute correlations for all pairs at once
    numerator = cp.sum(centered1 * centered2, axis=1)
    denom1 = cp.sum(centered1**2, axis=1)
    denom2 = cp.sum(centered2**2, axis=1)
    denominator = cp.sqrt(denom1 * denom2)

    # Avoid division by zero using vectorized where
    correlations = cp.where(denominator > 1e-10, numerator / denominator, 0.0)

    return correlations


# =============================================================================
# Single Frame Pair Detection Functions
# =============================================================================


def compute_pixel_difference_gpu(frame1: NDArray, frame2: NDArray) -> tuple[float, float]:
    """
    Compute pixel-level difference between two frames on GPU.

    This is the GPU-accelerated version of detection.compute_pixel_difference().
    Results are identical to the CPU version within 1e-5 tolerance.

    Args:
        frame1: First frame as NumPy array in BGR format, shape (H, W, 3).
        frame2: Second frame as NumPy array in BGR format, shape (H, W, 3).

    Returns:
        tuple: (mean_diff, changed_pixel_ratio)
            - mean_diff (float): Average pixel difference (0-255 scale).
            - changed_pixel_ratio (float): Ratio of significantly changed pixels (0.0-1.0).
    """
    cp = _get_cupy()

    # Transfer frames to GPU
    frame1_gpu = cp.asarray(frame1)
    frame2_gpu = cp.asarray(frame2)

    # Convert to grayscale
    gray1 = bgr_to_gray_gpu(frame1_gpu)
    gray2 = bgr_to_gray_gpu(frame2_gpu)

    # Compute absolute difference
    diff = cp.abs(gray1 - gray2)

    # Calculate mean difference
    mean_diff = float(cp.mean(diff).get())

    # Calculate changed pixel ratio (threshold = 30)
    threshold_mask = diff > 30
    changed_pixel_ratio = float(cp.sum(threshold_mask).get()) / diff.size

    return mean_diff, changed_pixel_ratio


def compute_histogram_distance_gpu(frame1: NDArray, frame2: NDArray) -> float:
    """
    Compute histogram-based distance between two frames on GPU.

    This is the GPU-accelerated version of detection.compute_histogram_distance().
    Results are identical to the CPU version within 1e-4 tolerance.

    Args:
        frame1: First frame as NumPy array in BGR format, shape (H, W, 3).
        frame2: Second frame as NumPy array in BGR format, shape (H, W, 3).

    Returns:
        float: Combined histogram distance (0-100 scale).
    """
    cp = _get_cupy()

    # Transfer frames to GPU
    frame1_gpu = cp.asarray(frame1)
    frame2_gpu = cp.asarray(frame2)

    # Convert to HSV
    hsv1 = bgr_to_hsv_gpu(frame1_gpu)
    hsv2 = bgr_to_hsv_gpu(frame2_gpu)

    # Compute histograms for each channel
    # H: 50 bins, range 0-180 (OpenCV convention)
    hist1_h = compute_histogram_gpu(hsv1[:, :, 0], bins=50, range_min=0, range_max=180)
    hist2_h = compute_histogram_gpu(hsv2[:, :, 0], bins=50, range_min=0, range_max=180)

    # S: 60 bins, range 0-256
    hist1_s = compute_histogram_gpu(hsv1[:, :, 1], bins=60, range_min=0, range_max=256)
    hist2_s = compute_histogram_gpu(hsv2[:, :, 1], bins=60, range_min=0, range_max=256)

    # V: 60 bins, range 0-256
    hist1_v = compute_histogram_gpu(hsv1[:, :, 2], bins=60, range_min=0, range_max=256)
    hist2_v = compute_histogram_gpu(hsv2[:, :, 2], bins=60, range_min=0, range_max=256)

    # Compute correlation for each channel and convert to distance
    corr_h = (1 - histogram_correlation_gpu(hist1_h, hist2_h)) * 100
    corr_s = (1 - histogram_correlation_gpu(hist1_s, hist2_s)) * 100
    corr_v = (1 - histogram_correlation_gpu(hist1_v, hist2_v)) * 100

    # Weighted combination (same as CPU version)
    combined_distance = (corr_h * 0.5) + (corr_s * 0.3) + (corr_v * 0.2)

    return combined_distance


# =============================================================================
# Batch Processing Functions
# =============================================================================


def compute_pixel_difference_batch_gpu(
    frames: list,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Compute pixel differences for a batch of consecutive frame pairs on GPU.

    This function processes multiple frames efficiently by keeping data on GPU
    memory and computing all differences in parallel.

    Supports both NumPy arrays (transferred to GPU) and CuPy arrays (already
    on GPU, e.g., from HardwareVideoReader with to_gpu=True). When frames are
    already on GPU, no CPU-GPU transfer is needed, eliminating the transfer
    bottleneck.

    Args:
        frames: List of N+1 frames in BGR format, each (H, W, 3).
                Can be NumPy arrays or CuPy arrays.
                Computes differences between consecutive pairs (N pairs total).

    Returns:
        tuple: (mean_diffs, changed_ratios)
            - mean_diffs: NumPy array of shape (N,) with mean differences.
            - changed_ratios: NumPy array of shape (N,) with changed pixel ratios.
    """
    cp = _get_cupy()

    if len(frames) < 2:
        return np.array([]), np.array([])

    # Stack frames - handles both NumPy and CuPy arrays automatically
    frames_gpu = _stack_frames_to_gpu(frames)

    # Convert all frames to grayscale at once
    grays = bgr_to_gray_gpu(frames_gpu)

    # Compute differences between consecutive frames
    diffs = cp.abs(grays[1:] - grays[:-1])

    # Calculate mean differences for all pairs
    mean_diffs = cp.mean(diffs, axis=(1, 2))

    # Calculate changed pixel ratios (threshold = 30)
    threshold_masks = diffs > 30
    pixels_per_frame = diffs.shape[1] * diffs.shape[2]
    changed_ratios = cp.sum(threshold_masks, axis=(1, 2)) / pixels_per_frame

    # Transfer results back to CPU
    return cp.asnumpy(mean_diffs), cp.asnumpy(changed_ratios)


def compute_histogram_distance_batch_gpu(frames: list) -> NDArray[np.floating]:
    """
    Compute histogram distances for a batch of consecutive frame pairs on GPU.

    This implementation processes all frames with optimized memory management:
    1. Single GPU upload for all frames (or direct use if already on GPU)
    2. Batch HSV conversion
    3. Histogram computation per frame (GPU-accelerated per histogram)
    4. Fully vectorized batch correlation (no GPU→CPU sync per pair)
    5. Single GPU→CPU transfer for final results

    Supports both NumPy arrays (transferred to GPU) and CuPy arrays (already
    on GPU, e.g., from HardwareVideoReader with to_gpu=True). When frames are
    already on GPU, no CPU-GPU transfer is needed, eliminating the transfer
    bottleneck.

    Args:
        frames: List of N+1 frames in BGR format, each (H, W, 3).
                Can be NumPy arrays or CuPy arrays.
                Computes distances between consecutive pairs (N pairs total).

    Returns:
        NumPy array of shape (N,) with combined histogram distances (0-100 scale).
    """
    cp = _get_cupy()

    if len(frames) < 2:
        return np.array([])

    # Stack frames - handles both NumPy and CuPy arrays automatically
    frames_gpu = _stack_frames_to_gpu(frames)

    # Convert all frames to HSV at once
    hsv_batch = bgr_to_hsv_gpu(frames_gpu)
    del frames_gpu  # Free GPU memory for original frames

    # Extract H, S, V channels for all frames: (N, H, W, 3) -> (N, H, W)
    h_channels = hsv_batch[:, :, :, 0]
    s_channels = hsv_batch[:, :, :, 1]
    v_channels = hsv_batch[:, :, :, 2]

    # Compute histograms (GPU-accelerated per histogram, loop for batch)
    h_hists = compute_histogram_batch_gpu(h_channels, bins=50, range_min=0, range_max=180)
    s_hists = compute_histogram_batch_gpu(s_channels, bins=60, range_min=0, range_max=256)
    v_hists = compute_histogram_batch_gpu(v_channels, bins=60, range_min=0, range_max=256)

    # Free HSV batch memory
    del hsv_batch, h_channels, s_channels, v_channels

    # Compute ALL correlations in batch (fully vectorized, no GPU→CPU sync per pair)
    corr_h = histogram_correlation_batch_gpu(h_hists[:-1], h_hists[1:])
    corr_s = histogram_correlation_batch_gpu(s_hists[:-1], s_hists[1:])
    corr_v = histogram_correlation_batch_gpu(v_hists[:-1], v_hists[1:])

    # Free histogram memory
    del h_hists, s_hists, v_hists

    # Compute weighted distances on GPU (same weights as CPU version)
    distances = (1 - corr_h) * 50 + (1 - corr_s) * 30 + (1 - corr_v) * 20

    # Single GPU→CPU transfer at the end
    result = cp.asnumpy(distances)

    # Cleanup correlation arrays
    del corr_h, corr_s, corr_v, distances

    return result


def free_gpu_memory() -> None:
    """
    Release all unused GPU memory from CuPy memory pools.

    Call this after processing large batches to free up GPU memory
    for other operations or to prevent memory fragmentation.
    """
    global _cp
    if _cp is not None:
        mempool = _cp.get_default_memory_pool()
        pinned_mempool = _cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
