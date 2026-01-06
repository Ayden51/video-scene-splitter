"""
Utility functions for file I/O, metrics analysis, and video processing.

This module contains helper functions for saving analysis data, computing
statistics, and managing output files.
"""

import os

import numpy as np


def save_metrics_to_csv(all_metrics, output_dir):
    """
    Save detailed frame-by-frame metrics to CSV file for analysis.

    This function exports all computed metrics to a CSV file and calculates
    statistical summaries. It also suggests an optimal threshold based on
    the video's characteristics using mean + 2 standard deviations.

    Args:
        all_metrics (list): List of dictionaries containing metrics for each frame.
            Each dictionary should have keys: 'frame', 'timestamp', 'hist_dist',
            'pixel_diff', 'changed_ratio'.
        output_dir (str): Directory where the CSV file will be saved.

    Returns:
        float: Suggested threshold value based on statistical analysis.

    Output Files:
        - cut_detection_metrics.csv: Frame-by-frame data with columns:
            Frame, Timestamp, Hist_Distance, Pixel_Diff, Changed_Ratio%

    Prints:
        - CSV file location
        - Statistical summary (min, max, mean, standard deviation)
        - Suggested threshold for optimal detection

    Note:
        The suggested threshold uses the formula: mean + 2*std_dev, which
        typically captures values that are 2 standard deviations above the
        normal variation, helping identify true scene changes.
    """
    csv_file = os.path.join(output_dir, "cut_detection_metrics.csv")

    # Write metrics to CSV
    with open(csv_file, "w") as f:
        f.write("Frame,Timestamp,Hist_Distance,Pixel_Diff,Changed_Ratio%\n")
        for m in all_metrics:
            f.write(
                f"{m['frame']},{m['timestamp']:.2f},{m['hist_dist']:.2f},"
                f"{m['pixel_diff']:.2f},{m['changed_ratio']:.2f}\n"
            )

    print(f"✓ Metrics saved to {csv_file}")

    # Calculate statistics
    hist_vals = [m["hist_dist"] for m in all_metrics]
    pixel_vals = [m["pixel_diff"] for m in all_metrics]

    print("\nStatistics:")
    print(
        f"  Histogram distance - Min: {min(hist_vals):.1f}, Max: {max(hist_vals):.1f}, "
        f"Mean: {np.mean(hist_vals):.1f}, StdDev: {np.std(hist_vals):.1f}"
    )
    print(
        f"  Pixel difference - Min: {min(pixel_vals):.1f}, Max: {max(pixel_vals):.1f}, "
        f"Mean: {np.mean(pixel_vals):.1f}, StdDev: {np.std(pixel_vals):.1f}"
    )

    # Suggest threshold
    suggested_threshold = np.mean(hist_vals) + 2 * np.std(hist_vals)
    print(f"  Suggested threshold: {suggested_threshold:.1f}")

    return suggested_threshold


def save_timestamps_to_file(
    video_path, scene_timestamps, threshold, output_dir, filename="timestamps.txt"
):
    """
    Save detected scene timestamps to a text file.

    Creates a human-readable text file containing all detected scene start
    times along with metadata about the video and detection parameters.
    Useful for reference, manual review, or importing into other tools.

    Args:
        video_path (str): Path to the original video file.
        scene_timestamps (list): List of scene start times in seconds.
        threshold (float): Detection threshold that was used.
        output_dir (str): Directory where the file will be saved.
        filename (str, optional): Name of the output file.
            Defaults to "timestamps.txt".

    Output Format:
        Video: {video_path}
        Total scenes: {count}
        Threshold: {threshold}

        Scene 1: 0.00s
        Scene 2: 5.23s
        Scene 3: 12.45s
        ...

    Side Effects:
        - Creates a text file in output_dir
        - Prints confirmation message with file path

    Example:
        >>> save_timestamps_to_file("video.mp4", [0.0, 5.23], 20.0, "output")
        ✓ Timestamps saved to output/timestamps.txt
    """
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w") as f:
        f.write(f"Video: {video_path}\n")
        f.write(f"Total scenes: {len(scene_timestamps)}\n")
        f.write(f"Threshold: {threshold}\n\n")
        for i, ts in enumerate(scene_timestamps):
            f.write(f"Scene {i + 1}: {ts:.2f}s\n")
    print(f"✓ Timestamps saved to {output_path}")


def save_debug_frames(
    output_dir, scene_number, frame_number_before, frame_number_after, frame_before, frame_after
):
    """
    Save before and after frames at a detected cut point for debugging.

    Args:
        output_dir (str): Directory where frames will be saved.
        scene_number (int): Scene number (for filename).
        frame_number_before (int): Frame number before the cut.
        frame_number_after (int): Frame number after the cut.
        frame_before (numpy.ndarray): Frame image before the cut.
        frame_after (numpy.ndarray): Frame image after the cut.

    Output Files:
        - cut_{scene:03d}_before_f{frame}.jpg
        - cut_{scene:03d}_after_f{frame}.jpg
    """
    import cv2

    # Save the frame before cut
    before_path = os.path.join(
        output_dir, f"cut_{scene_number:03d}_before_f{frame_number_before}.jpg"
    )
    cv2.imwrite(before_path, frame_before)

    # Save the frame after cut
    after_path = os.path.join(output_dir, f"cut_{scene_number:03d}_after_f{frame_number_after}.jpg")
    cv2.imwrite(after_path, frame_after)
