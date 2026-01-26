"""
Scene detection algorithms for identifying hard cuts in video content.

This module contains the core computer vision algorithms used to detect
scene changes by comparing consecutive video frames.
"""

import cv2
import numpy as np


def compute_histogram_distance(frame1, frame2):
    """
    Compute histogram-based distance between two consecutive frames.

    This function converts frames to HSV color space and calculates histograms
    for each channel (Hue, Saturation, Value). It then compares these histograms
    using correlation to determine how different the color content is between frames.
    This approach is excellent for detecting hard cuts where the entire image
    content changes abruptly.

    Args:
        frame1 (numpy.ndarray): First frame in BGR format (OpenCV default).
            Shape: (height, width, 3)
        frame2 (numpy.ndarray): Second frame in BGR format.
            Shape: (height, width, 3)

    Returns:
        float: Combined histogram distance on a 0-100 scale.
            - 0: Frames are identical
            - 100: Frames are completely different
            - Typical values for same scene: 5-15
            - Typical values for hard cuts: 30-80

    Note:
        Uses weighted combination: Hue (50%), Saturation (30%), Value (20%).
        HSV color space is used because it separates color information (H, S)
        from brightness (V), making it more robust to lighting changes.
    """
    # Convert to HSV (better for content comparison than RGB)
    # HSV separates color from brightness, making detection more robust
    hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)

    # Calculate histograms for each channel
    # Hue: 50 bins (0-180 degrees in OpenCV)
    hist1_h = cv2.calcHist([hsv1], [0], None, [50], [0, 180])
    hist2_h = cv2.calcHist([hsv2], [0], None, [50], [0, 180])

    # Saturation and Value: 60 bins each (0-256)
    hist1_s = cv2.calcHist([hsv1], [1], None, [60], [0, 256])
    hist2_s = cv2.calcHist([hsv2], [1], None, [60], [0, 256])

    hist1_v = cv2.calcHist([hsv1], [2], None, [60], [0, 256])
    hist2_v = cv2.calcHist([hsv2], [2], None, [60], [0, 256])

    # Normalize histograms to 0-1 range for fair comparison
    cv2.normalize(hist1_h, hist1_h, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist1_s, hist1_s, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist1_v, hist1_v, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2_h, hist2_h, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2_s, hist2_s, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2_v, hist2_v, 0, 1, cv2.NORM_MINMAX)

    # Use correlation method (returns 0-1, where 0 = different, 1 = same)
    # Convert to 0-100 scale where higher = more different
    corr_h = (1 - cv2.compareHist(hist1_h, hist2_h, cv2.HISTCMP_CORREL)) * 100
    corr_s = (1 - cv2.compareHist(hist1_s, hist2_s, cv2.HISTCMP_CORREL)) * 100
    corr_v = (1 - cv2.compareHist(hist1_v, hist2_v, cv2.HISTCMP_CORREL)) * 100

    # Weight hue and saturation more (color content changes are more important)
    combined_distance = (corr_h * 0.5) + (corr_s * 0.3) + (corr_v * 0.2)

    return combined_distance


def compute_pixel_difference(frame1, frame2):
    """
    Compute direct pixel-level difference between two consecutive frames.

    This function converts frames to grayscale and calculates the absolute
    difference between corresponding pixels. It provides two metrics:
    mean difference and the ratio of significantly changed pixels.
    This approach is particularly good for detecting instant, dramatic changes.

    Args:
        frame1 (numpy.ndarray): First frame in BGR format.
            Shape: (height, width, 3)
        frame2 (numpy.ndarray): Second frame in BGR format.
            Shape: (height, width, 3)

    Returns:
        tuple: A tuple containing:
            - mean_diff (float): Average pixel difference across the entire frame.
                Range: 0-255 (grayscale intensity scale)
                - 0: Frames are identical
                - Typical same scene: 5-15
                - Typical hard cut: 30-100
            - changed_pixel_ratio (float): Ratio of pixels that changed significantly.
                Range: 0.0-1.0 (0% to 100%)
                - 0.0: No pixels changed significantly
                - 0.2: 20% of pixels changed (typical threshold for hard cuts)
                - 1.0: All pixels changed significantly

    Note:
        A pixel is considered "significantly changed" if its grayscale value
        differs by more than 30 (on a 0-255 scale). This threshold helps
        filter out minor variations due to compression or noise.
    """
    # Convert to grayscale for simpler, faster comparison
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate absolute difference between all pixels
    diff = cv2.absdiff(gray1, gray2)
    mean_diff = np.mean(diff)

    # Calculate percentage of significantly changed pixels
    # Threshold of 30 filters out minor variations from compression/noise
    threshold_mask = diff > 30  # Pixels that changed significantly
    changed_pixel_ratio = np.sum(threshold_mask) / diff.size

    return mean_diff, changed_pixel_ratio


def is_hard_cut(hist_dist, pixel_diff, changed_ratio, threshold, frames_since_last, min_frame_gap):
    """
    Determine if a hard cut occurred based on multiple metrics.

    A hard cut is detected when ALL of the following conditions are met:
    1. Histogram distance exceeds threshold (color content changed)
    2. Pixel difference exceeds threshold (brightness/intensity changed)
    3. Changed pixel ratio exceeds 20% (spatial coverage of change)
    4. Sufficient time has passed since the last detected cut

    Args:
        hist_dist (float): Histogram distance metric (0-100 scale).
        pixel_diff (float): Mean pixel difference (0-255 scale).
        changed_ratio (float): Ratio of significantly changed pixels (0.0-1.0).
        threshold (float): Detection threshold for hist_dist and pixel_diff.
        frames_since_last (int): Number of frames since last detected cut.
        min_frame_gap (int): Minimum required frames between cuts.

    Returns:
        bool: True if all hard cut conditions are met, False otherwise.
    """
    return (
        hist_dist > threshold
        and pixel_diff > threshold
        and changed_ratio > 0.2
        and frames_since_last >= min_frame_gap
    )


def compute_pixel_difference_batch_cpu(frames: list) -> list[tuple[float, float]]:
    """
    Compute pixel differences for a batch of consecutive frames on CPU.

    This is the CPU implementation for batch processing, used in hybrid mode
    when GPU is not beneficial (e.g., for SD content or when GPU is unavailable).

    Args:
        frames: List of frames in BGR format. Must have at least 2 frames.
            Each frame shape: (height, width, 3)

    Returns:
        List of (mean_diff, changed_ratio) tuples for each consecutive frame pair.
        Length will be len(frames) - 1.

    Example:
        >>> frames = [frame1, frame2, frame3]
        >>> results = compute_pixel_difference_batch_cpu(frames)
        >>> # results[0] = difference between frame1 and frame2
        >>> # results[1] = difference between frame2 and frame3
    """
    if len(frames) < 2:
        return []

    results = []
    for i in range(len(frames) - 1):
        mean_diff, changed_ratio = compute_pixel_difference(frames[i], frames[i + 1])
        results.append((mean_diff, changed_ratio))

    return results


def compute_histogram_distance_batch_cpu(frames: list) -> list[float]:
    """
    Compute histogram distances for a batch of consecutive frames on CPU.

    This is the CPU implementation for batch processing, used in hybrid mode.
    Based on Phase 2A/2B benchmarks, CPU histogram is 1.35x faster than GPU
    due to transfer overhead, so this is the preferred method for histogram
    computation even when GPU is available.

    Args:
        frames: List of frames in BGR format. Must have at least 2 frames.
            Each frame shape: (height, width, 3)

    Returns:
        List of histogram distances for each consecutive frame pair.
        Length will be len(frames) - 1.

    Example:
        >>> frames = [frame1, frame2, frame3]
        >>> distances = compute_histogram_distance_batch_cpu(frames)
        >>> # distances[0] = histogram distance between frame1 and frame2
        >>> # distances[1] = histogram distance between frame2 and frame3
    """
    if len(frames) < 2:
        return []

    distances = []
    for i in range(len(frames) - 1):
        dist = compute_histogram_distance(frames[i], frames[i + 1])
        distances.append(dist)

    return distances
