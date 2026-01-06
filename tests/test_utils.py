"""
Unit tests for video_scene_splitter.utils module.

Tests the utility functions for file I/O, metrics analysis, and debugging.
"""

import os

import cv2
import numpy as np
import pytest

from video_scene_splitter.utils import (
    save_debug_frames,
    save_metrics_to_csv,
    save_timestamps_to_file,
)


class TestSaveMetricsToCSV:
    """Tests for save_metrics_to_csv function."""

    def test_creates_csv_file(self, temp_output_dir):
        """Should create a CSV file in the output directory."""
        metrics = [
            {
                "frame": 1,
                "timestamp": 0.03,
                "hist_dist": 5.2,
                "pixel_diff": 12.4,
                "changed_ratio": 3.2,
            },
            {
                "frame": 2,
                "timestamp": 0.07,
                "hist_dist": 6.1,
                "pixel_diff": 13.8,
                "changed_ratio": 3.4,
            },
        ]
        save_metrics_to_csv(metrics, temp_output_dir)

        csv_path = os.path.join(temp_output_dir, "cut_detection_metrics.csv")
        assert os.path.exists(csv_path)

    def test_csv_has_correct_header(self, temp_output_dir):
        """CSV file should have the correct header row."""
        metrics = [
            {
                "frame": 1,
                "timestamp": 0.03,
                "hist_dist": 5.2,
                "pixel_diff": 12.4,
                "changed_ratio": 3.2,
            },
        ]
        save_metrics_to_csv(metrics, temp_output_dir)

        csv_path = os.path.join(temp_output_dir, "cut_detection_metrics.csv")
        with open(csv_path) as f:
            header = f.readline().strip()
        assert header == "Frame,Timestamp,Hist_Distance,Pixel_Diff,Changed_Ratio%"

    def test_csv_contains_all_metrics(self, temp_output_dir):
        """CSV file should contain all provided metrics."""
        metrics = [
            {
                "frame": 1,
                "timestamp": 0.03,
                "hist_dist": 5.2,
                "pixel_diff": 12.4,
                "changed_ratio": 3.2,
            },
            {
                "frame": 2,
                "timestamp": 0.07,
                "hist_dist": 6.1,
                "pixel_diff": 13.8,
                "changed_ratio": 3.4,
            },
        ]
        save_metrics_to_csv(metrics, temp_output_dir)

        csv_path = os.path.join(temp_output_dir, "cut_detection_metrics.csv")
        with open(csv_path) as f:
            lines = f.readlines()
        # Header + 2 data rows
        assert len(lines) == 3

    def test_returns_suggested_threshold(self, temp_output_dir):
        """Should return a suggested threshold value."""
        metrics = [
            {
                "frame": i,
                "timestamp": i * 0.03,
                "hist_dist": 10.0 + i,
                "pixel_diff": 15.0 + i,
                "changed_ratio": 2.0,
            }
            for i in range(10)
        ]
        threshold = save_metrics_to_csv(metrics, temp_output_dir)
        assert isinstance(threshold, (float, np.floating))
        assert threshold > 0

    def test_suggested_threshold_calculation(self, temp_output_dir):
        """Suggested threshold should be mean + 2*std_dev."""
        # Create metrics with known values
        hist_values = [10.0, 10.0, 10.0, 10.0, 10.0]  # Mean=10, StdDev=0
        metrics = [
            {
                "frame": i,
                "timestamp": i * 0.03,
                "hist_dist": val,
                "pixel_diff": 15.0,
                "changed_ratio": 2.0,
            }
            for i, val in enumerate(hist_values)
        ]
        threshold = save_metrics_to_csv(metrics, temp_output_dir)
        # With StdDev=0, threshold should be approximately mean (10.0)
        assert threshold == pytest.approx(10.0, abs=0.1)

    def test_handles_empty_metrics_list(self, temp_output_dir):
        """Should handle empty metrics list gracefully."""
        metrics = []
        with pytest.raises((ValueError, ZeroDivisionError)):
            save_metrics_to_csv(metrics, temp_output_dir)

    def test_csv_formatting(self, temp_output_dir):
        """CSV values should be formatted with correct precision."""
        metrics = [
            {
                "frame": 1,
                "timestamp": 0.033333,
                "hist_dist": 5.234,
                "pixel_diff": 12.456,
                "changed_ratio": 3.289,
            },
        ]
        save_metrics_to_csv(metrics, temp_output_dir)

        csv_path = os.path.join(temp_output_dir, "cut_detection_metrics.csv")
        with open(csv_path) as f:
            f.readline()  # Skip header
            data_line = f.readline().strip()
        # Check that values are formatted to 2 decimal places
        assert "0.03" in data_line
        assert "5.23" in data_line
        assert "12.46" in data_line


class TestSaveTimestampsToFile:
    """Tests for save_timestamps_to_file function."""

    def test_creates_timestamp_file(self, temp_output_dir):
        """Should create a timestamp file in the output directory."""
        timestamps = [0.0, 5.23, 12.45]
        save_timestamps_to_file("test_video.mp4", timestamps, 20.0, temp_output_dir)

        timestamp_path = os.path.join(temp_output_dir, "timestamps.txt")
        assert os.path.exists(timestamp_path)

    def test_custom_filename(self, temp_output_dir):
        """Should support custom filename."""
        timestamps = [0.0, 5.23]
        save_timestamps_to_file("test_video.mp4", timestamps, 20.0, temp_output_dir, "custom.txt")

        custom_path = os.path.join(temp_output_dir, "custom.txt")
        assert os.path.exists(custom_path)

    def test_file_contains_metadata(self, temp_output_dir):
        """File should contain video path, scene count, and threshold."""
        timestamps = [0.0, 5.23, 12.45]
        save_timestamps_to_file("test_video.mp4", timestamps, 20.0, temp_output_dir)

        timestamp_path = os.path.join(temp_output_dir, "timestamps.txt")
        with open(timestamp_path) as f:
            content = f.read()

        assert "test_video.mp4" in content
        assert "Total scenes: 3" in content
        assert "Threshold: 20.0" in content

    def test_file_contains_all_timestamps(self, temp_output_dir):
        """File should contain all scene timestamps."""
        timestamps = [0.0, 5.23, 12.45, 20.67]
        save_timestamps_to_file("test_video.mp4", timestamps, 20.0, temp_output_dir)

        timestamp_path = os.path.join(temp_output_dir, "timestamps.txt")
        with open(timestamp_path) as f:
            content = f.read()

        assert "Scene 1: 0.00s" in content
        assert "Scene 2: 5.23s" in content
        assert "Scene 3: 12.45s" in content
        assert "Scene 4: 20.67s" in content

    def test_handles_empty_timestamps(self, temp_output_dir):
        """Should handle empty timestamps list."""
        timestamps = []
        save_timestamps_to_file("test_video.mp4", timestamps, 20.0, temp_output_dir)

        timestamp_path = os.path.join(temp_output_dir, "timestamps.txt")
        with open(timestamp_path) as f:
            content = f.read()

        assert "Total scenes: 0" in content

    def test_timestamp_formatting(self, temp_output_dir):
        """Timestamps should be formatted to 2 decimal places."""
        timestamps = [0.0, 5.234567, 12.456789]
        save_timestamps_to_file("test_video.mp4", timestamps, 20.0, temp_output_dir)

        timestamp_path = os.path.join(temp_output_dir, "timestamps.txt")
        with open(timestamp_path) as f:
            content = f.read()

        assert "5.23s" in content
        assert "12.46s" in content


class TestSaveDebugFrames:
    """Tests for save_debug_frames function."""

    def test_saves_before_frame(self, temp_output_dir, black_frame):
        """Should save the 'before' frame."""
        save_debug_frames(temp_output_dir, 1, 100, 101, black_frame, black_frame)

        before_path = os.path.join(temp_output_dir, "cut_001_before_f100.jpg")
        assert os.path.exists(before_path)

    def test_saves_after_frame(self, temp_output_dir, black_frame):
        """Should save the 'after' frame."""
        save_debug_frames(temp_output_dir, 1, 100, 101, black_frame, black_frame)

        after_path = os.path.join(temp_output_dir, "cut_001_after_f101.jpg")
        assert os.path.exists(after_path)

    def test_saves_both_frames(self, temp_output_dir, black_frame, white_frame):
        """Should save both before and after frames."""
        save_debug_frames(temp_output_dir, 2, 200, 201, black_frame, white_frame)

        before_path = os.path.join(temp_output_dir, "cut_002_before_f200.jpg")
        after_path = os.path.join(temp_output_dir, "cut_002_after_f201.jpg")

        assert os.path.exists(before_path)
        assert os.path.exists(after_path)

    def test_scene_number_formatting(self, temp_output_dir, black_frame):
        """Scene number should be zero-padded to 3 digits."""
        save_debug_frames(temp_output_dir, 5, 100, 101, black_frame, black_frame)

        before_path = os.path.join(temp_output_dir, "cut_005_before_f100.jpg")
        assert os.path.exists(before_path)

    def test_saved_images_are_valid(self, temp_output_dir, red_frame, blue_frame):
        """Saved images should be valid and readable."""
        save_debug_frames(temp_output_dir, 1, 100, 101, red_frame, blue_frame)

        before_path = os.path.join(temp_output_dir, "cut_001_before_f100.jpg")
        after_path = os.path.join(temp_output_dir, "cut_001_after_f101.jpg")

        # Try to read the saved images
        before_img = cv2.imread(before_path)
        after_img = cv2.imread(after_path)

        assert before_img is not None
        assert after_img is not None
        assert before_img.shape == red_frame.shape
        assert after_img.shape == blue_frame.shape

    def test_handles_large_scene_numbers(self, temp_output_dir, black_frame):
        """Should handle large scene numbers correctly."""
        save_debug_frames(temp_output_dir, 999, 10000, 10001, black_frame, black_frame)

        before_path = os.path.join(temp_output_dir, "cut_999_before_f10000.jpg")
        assert os.path.exists(before_path)

    def test_different_frame_sizes(self, temp_output_dir):
        """Should handle frames of different sizes."""
        small_frame = np.zeros((50, 50, 3), dtype=np.uint8)
        large_frame = np.zeros((200, 200, 3), dtype=np.uint8)

        save_debug_frames(temp_output_dir, 1, 100, 101, small_frame, large_frame)

        before_path = os.path.join(temp_output_dir, "cut_001_before_f100.jpg")
        after_path = os.path.join(temp_output_dir, "cut_001_after_f101.jpg")

        before_img = cv2.imread(before_path)
        after_img = cv2.imread(after_path)

        assert before_img.shape[:2] == (50, 50)
        assert after_img.shape[:2] == (200, 200)
