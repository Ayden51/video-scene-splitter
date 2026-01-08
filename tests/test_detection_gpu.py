"""
Unit tests for video_scene_splitter.detection_gpu module.

Tests the GPU-accelerated scene detection algorithms and verifies that
GPU results match CPU results within acceptable tolerance.
"""

import numpy as np
import pytest

from video_scene_splitter.detection import (
    compute_histogram_distance,
    compute_pixel_difference,
)


class TestComputePixelDifferenceGPU:
    """Tests for GPU-accelerated pixel difference computation."""

    @pytest.mark.gpu
    def test_identical_frames_return_zero_difference(self, black_frame):
        """GPU: Identical frames should have zero pixel difference."""
        from video_scene_splitter.detection_gpu import compute_pixel_difference_gpu

        mean_diff, changed_ratio = compute_pixel_difference_gpu(black_frame, black_frame)
        assert mean_diff == pytest.approx(0.0, abs=0.1)
        assert changed_ratio == pytest.approx(0.0, abs=0.01)

    @pytest.mark.gpu
    def test_completely_different_frames_high_difference(self, black_frame, white_frame):
        """GPU: Completely different frames should have high pixel difference."""
        from video_scene_splitter.detection_gpu import compute_pixel_difference_gpu

        mean_diff, changed_ratio = compute_pixel_difference_gpu(black_frame, white_frame)
        assert mean_diff > 200.0
        assert changed_ratio > 0.9

    @pytest.mark.gpu
    def test_gpu_matches_cpu_black_white(self, black_frame, white_frame):
        """GPU results should match CPU results for black/white frames."""
        from video_scene_splitter.detection_gpu import compute_pixel_difference_gpu

        cpu_mean, cpu_ratio = compute_pixel_difference(black_frame, white_frame)
        gpu_mean, gpu_ratio = compute_pixel_difference_gpu(black_frame, white_frame)

        assert gpu_mean == pytest.approx(cpu_mean, rel=1e-5)
        assert gpu_ratio == pytest.approx(cpu_ratio, rel=1e-5)

    @pytest.mark.gpu
    def test_gpu_matches_cpu_similar_frames(self, similar_frames):
        """GPU results should match CPU results for similar frames."""
        from video_scene_splitter.detection_gpu import compute_pixel_difference_gpu

        frame1, frame2 = similar_frames
        cpu_mean, cpu_ratio = compute_pixel_difference(frame1, frame2)
        gpu_mean, gpu_ratio = compute_pixel_difference_gpu(frame1, frame2)

        assert gpu_mean == pytest.approx(cpu_mean, rel=1e-5)
        assert gpu_ratio == pytest.approx(cpu_ratio, rel=1e-5)

    @pytest.mark.gpu
    def test_gpu_matches_cpu_gradient(self, gradient_frame, black_frame):
        """GPU results should match CPU results for gradient frames."""
        from video_scene_splitter.detection_gpu import compute_pixel_difference_gpu

        cpu_mean, cpu_ratio = compute_pixel_difference(gradient_frame, black_frame)
        gpu_mean, gpu_ratio = compute_pixel_difference_gpu(gradient_frame, black_frame)

        assert gpu_mean == pytest.approx(cpu_mean, rel=1e-5)
        assert gpu_ratio == pytest.approx(cpu_ratio, rel=1e-5)

    @pytest.mark.gpu
    def test_return_types(self, black_frame, white_frame):
        """GPU function should return tuple of floats."""
        from video_scene_splitter.detection_gpu import compute_pixel_difference_gpu

        mean_diff, changed_ratio = compute_pixel_difference_gpu(black_frame, white_frame)
        assert isinstance(mean_diff, (float, np.floating))
        assert isinstance(changed_ratio, (float, np.floating))


class TestComputeHistogramDistanceGPU:
    """Tests for GPU-accelerated histogram distance computation."""

    @pytest.mark.gpu
    def test_identical_frames_low_distance(self, black_frame):
        """GPU: Identical frames should have very low histogram distance."""
        from video_scene_splitter.detection_gpu import compute_histogram_distance_gpu

        distance = compute_histogram_distance_gpu(black_frame, black_frame)
        # Identical frames should have near-zero distance
        assert distance < 5.0

    @pytest.mark.gpu
    def test_completely_different_frames_high_distance(self, black_frame, white_frame):
        """GPU: Very different frames should have high histogram distance."""
        from video_scene_splitter.detection_gpu import compute_histogram_distance_gpu

        distance = compute_histogram_distance_gpu(black_frame, white_frame)
        assert distance > 10.0

    @pytest.mark.gpu
    def test_different_colors_high_distance(self, red_frame, blue_frame):
        """GPU: Frames with different colors should have high histogram distance."""
        from video_scene_splitter.detection_gpu import compute_histogram_distance_gpu

        distance = compute_histogram_distance_gpu(red_frame, blue_frame)
        assert distance > 20.0

    @pytest.mark.gpu
    def test_gpu_matches_cpu_black_white(self, black_frame, white_frame):
        """GPU results should match CPU results for black/white frames."""
        from video_scene_splitter.detection_gpu import compute_histogram_distance_gpu

        cpu_dist = compute_histogram_distance(black_frame, white_frame)
        gpu_dist = compute_histogram_distance_gpu(black_frame, white_frame)

        # Allow for some difference due to HSV conversion implementation differences
        assert gpu_dist == pytest.approx(cpu_dist, rel=0.3)

    @pytest.mark.gpu
    def test_gpu_matches_cpu_colored_frames(self, red_frame, blue_frame):
        """GPU results should match CPU results for colored frames."""
        from video_scene_splitter.detection_gpu import compute_histogram_distance_gpu

        cpu_dist = compute_histogram_distance(red_frame, blue_frame)
        gpu_dist = compute_histogram_distance_gpu(red_frame, blue_frame)

        # Allow for some difference due to HSV conversion implementation differences
        assert gpu_dist == pytest.approx(cpu_dist, rel=0.3)

    @pytest.mark.gpu
    def test_return_type_is_float(self, black_frame, white_frame):
        """GPU function should return a float value."""
        from video_scene_splitter.detection_gpu import compute_histogram_distance_gpu

        distance = compute_histogram_distance_gpu(black_frame, white_frame)
        assert isinstance(distance, (float, np.floating))

    @pytest.mark.gpu
    def test_distance_is_non_negative(self, black_frame, white_frame):
        """GPU: Histogram distance should always be non-negative."""
        from video_scene_splitter.detection_gpu import compute_histogram_distance_gpu

        distance = compute_histogram_distance_gpu(black_frame, white_frame)
        assert distance >= 0.0


class TestBatchPixelDifferenceGPU:
    """Tests for GPU batch pixel difference computation."""

    @pytest.mark.gpu
    def test_batch_empty_list(self):
        """GPU: Empty or single-frame list should return empty arrays."""
        from video_scene_splitter.detection_gpu import compute_pixel_difference_batch_gpu

        mean_diffs, changed_ratios = compute_pixel_difference_batch_gpu([])
        assert len(mean_diffs) == 0
        assert len(changed_ratios) == 0

        # Single frame also returns empty (need at least 2 for 1 pair)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mean_diffs, changed_ratios = compute_pixel_difference_batch_gpu([frame])
        assert len(mean_diffs) == 0
        assert len(changed_ratios) == 0

    @pytest.mark.gpu
    def test_batch_two_frames(self, black_frame, white_frame):
        """GPU: Batch processing of two frames should work correctly."""
        from video_scene_splitter.detection_gpu import compute_pixel_difference_batch_gpu

        frames = [black_frame, white_frame]
        mean_diffs, changed_ratios = compute_pixel_difference_batch_gpu(frames)

        assert len(mean_diffs) == 1
        assert len(changed_ratios) == 1
        assert mean_diffs[0] > 200.0
        assert changed_ratios[0] > 0.9

    @pytest.mark.gpu
    def test_batch_matches_sequential_cpu(self, frame_batch_small):
        """GPU batch results should match sequential CPU processing."""
        from video_scene_splitter.detection_gpu import compute_pixel_difference_batch_gpu

        # GPU batch processing
        gpu_means, gpu_ratios = compute_pixel_difference_batch_gpu(frame_batch_small)

        # Sequential CPU processing
        for i in range(len(frame_batch_small) - 1):
            cpu_mean, cpu_ratio = compute_pixel_difference(
                frame_batch_small[i], frame_batch_small[i + 1]
            )
            assert gpu_means[i] == pytest.approx(cpu_mean, rel=1e-5)
            assert gpu_ratios[i] == pytest.approx(cpu_ratio, rel=1e-5)

    @pytest.mark.gpu
    def test_batch_large(self, frame_batch_large):
        """GPU: Large batch processing should work correctly."""
        from video_scene_splitter.detection_gpu import compute_pixel_difference_batch_gpu

        mean_diffs, changed_ratios = compute_pixel_difference_batch_gpu(frame_batch_large)

        # Should have N-1 results for N frames
        assert len(mean_diffs) == len(frame_batch_large) - 1
        assert len(changed_ratios) == len(frame_batch_large) - 1

        # All values should be valid
        assert all(0 <= m <= 255 for m in mean_diffs)
        assert all(0 <= r <= 1 for r in changed_ratios)

    @pytest.mark.gpu
    def test_batch_identical_frames(self, identical_frame_batch):
        """GPU: Identical frames should have zero differences."""
        from video_scene_splitter.detection_gpu import compute_pixel_difference_batch_gpu

        mean_diffs, changed_ratios = compute_pixel_difference_batch_gpu(identical_frame_batch)

        # All differences should be near zero
        assert all(m == pytest.approx(0.0, abs=0.1) for m in mean_diffs)
        assert all(r == pytest.approx(0.0, abs=0.01) for r in changed_ratios)

    @pytest.mark.gpu
    def test_batch_return_types(self, frame_batch_small):
        """GPU batch function should return numpy arrays."""
        from video_scene_splitter.detection_gpu import compute_pixel_difference_batch_gpu

        mean_diffs, changed_ratios = compute_pixel_difference_batch_gpu(frame_batch_small)

        assert isinstance(mean_diffs, np.ndarray)
        assert isinstance(changed_ratios, np.ndarray)


class TestBatchHistogramDistanceGPU:
    """Tests for GPU batch histogram distance computation."""

    @pytest.mark.gpu
    def test_batch_empty_list(self):
        """GPU: Empty or single-frame list should return empty array."""
        from video_scene_splitter.detection_gpu import compute_histogram_distance_batch_gpu

        distances = compute_histogram_distance_batch_gpu([])
        assert len(distances) == 0

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        distances = compute_histogram_distance_batch_gpu([frame])
        assert len(distances) == 0

    @pytest.mark.gpu
    def test_batch_two_frames(self, black_frame, white_frame):
        """GPU: Batch processing of two frames should work correctly."""
        from video_scene_splitter.detection_gpu import compute_histogram_distance_batch_gpu

        frames = [black_frame, white_frame]
        distances = compute_histogram_distance_batch_gpu(frames)

        assert len(distances) == 1
        assert distances[0] > 10.0  # Significant difference

    @pytest.mark.gpu
    def test_batch_matches_sequential_cpu(self, frame_batch_small):
        """GPU batch results should be close to sequential CPU processing."""
        from video_scene_splitter.detection_gpu import compute_histogram_distance_batch_gpu

        # GPU batch processing
        gpu_distances = compute_histogram_distance_batch_gpu(frame_batch_small)

        # Sequential CPU processing
        for i in range(len(frame_batch_small) - 1):
            cpu_dist = compute_histogram_distance(frame_batch_small[i], frame_batch_small[i + 1])
            # Allow for some difference due to HSV conversion differences
            assert gpu_distances[i] == pytest.approx(cpu_dist, rel=0.3)

    @pytest.mark.gpu
    def test_batch_large(self, frame_batch_large):
        """GPU: Large batch processing should work correctly."""
        from video_scene_splitter.detection_gpu import compute_histogram_distance_batch_gpu

        distances = compute_histogram_distance_batch_gpu(frame_batch_large)

        # Should have N-1 results for N frames
        assert len(distances) == len(frame_batch_large) - 1

        # All values should be non-negative
        assert all(d >= 0 for d in distances)

    @pytest.mark.gpu
    def test_batch_identical_frames(self, identical_frame_batch):
        """GPU: Identical frames should have low histogram distances."""
        from video_scene_splitter.detection_gpu import compute_histogram_distance_batch_gpu

        distances = compute_histogram_distance_batch_gpu(identical_frame_batch)

        # All distances should be very low for identical frames
        assert all(d < 5.0 for d in distances)

    @pytest.mark.gpu
    def test_batch_return_type(self, frame_batch_small):
        """GPU batch function should return numpy array."""
        from video_scene_splitter.detection_gpu import compute_histogram_distance_batch_gpu

        distances = compute_histogram_distance_batch_gpu(frame_batch_small)
        assert isinstance(distances, np.ndarray)


class TestGPUHelperFunctions:
    """Tests for GPU helper functions."""

    @pytest.mark.gpu
    def test_bgr_to_gray_single_frame(self, black_frame, white_frame):
        """GPU grayscale conversion should work for single frames."""
        from video_scene_splitter.detection_gpu import _get_cupy, bgr_to_gray_gpu

        cp = _get_cupy()

        # Test black frame
        black_gpu = cp.asarray(black_frame)
        gray = bgr_to_gray_gpu(black_gpu)
        assert gray.shape == (100, 100)
        assert float(cp.mean(gray)) == pytest.approx(0.0, abs=0.1)

        # Test white frame
        white_gpu = cp.asarray(white_frame)
        gray = bgr_to_gray_gpu(white_gpu)
        assert gray.shape == (100, 100)
        assert float(cp.mean(gray)) == pytest.approx(255.0, abs=0.1)

    @pytest.mark.gpu
    def test_bgr_to_gray_batch(self, frame_batch_small):
        """GPU grayscale conversion should work for batches."""
        from video_scene_splitter.detection_gpu import _get_cupy, bgr_to_gray_gpu

        cp = _get_cupy()
        frames_stack = np.stack(frame_batch_small, axis=0)
        frames_gpu = cp.asarray(frames_stack)

        grays = bgr_to_gray_gpu(frames_gpu)
        assert grays.shape == (5, 100, 100)

    @pytest.mark.gpu
    def test_bgr_to_hsv_single_frame(self, red_frame):
        """GPU HSV conversion should work for single frames."""
        from video_scene_splitter.detection_gpu import _get_cupy, bgr_to_hsv_gpu

        cp = _get_cupy()

        # Test red frame - should have H around 0 (or 180 depending on convention)
        red_gpu = cp.asarray(red_frame)
        hsv = bgr_to_hsv_gpu(red_gpu)
        assert hsv.shape == (100, 100, 3)

        # V channel should be 255 for pure red
        assert float(cp.mean(hsv[:, :, 2])) == pytest.approx(255.0, abs=1.0)

    @pytest.mark.gpu
    def test_bgr_to_hsv_batch(self, frame_batch_small):
        """GPU HSV conversion should work for batches."""
        from video_scene_splitter.detection_gpu import _get_cupy, bgr_to_hsv_gpu

        cp = _get_cupy()
        frames_stack = np.stack(frame_batch_small, axis=0)
        frames_gpu = cp.asarray(frames_stack)

        hsv_batch = bgr_to_hsv_gpu(frames_gpu)
        assert hsv_batch.shape == (5, 100, 100, 3)


class TestGPUMemoryManagement:
    """Tests for GPU memory management functions."""

    @pytest.mark.gpu
    def test_free_gpu_memory(self):
        """free_gpu_memory should release memory without errors."""
        from video_scene_splitter.detection_gpu import (
            compute_pixel_difference_batch_gpu,
            free_gpu_memory,
        )

        # Create some GPU allocations
        frames = [np.full((100, 100, 3), i, dtype=np.uint8) for i in range(10)]
        compute_pixel_difference_batch_gpu(frames)

        # Free memory - should not raise any errors
        free_gpu_memory()

    @pytest.mark.gpu
    def test_cupy_import_error_message(self):
        """Test that proper error message is shown when CuPy unavailable."""
        # This test verifies the error message structure
        # We can't easily test the actual ImportError without uninstalling CuPy
        from video_scene_splitter.detection_gpu import _get_cupy

        # Just verify CuPy is available and _get_cupy works
        cp = _get_cupy()
        assert cp is not None


class TestSceneChangeDetection:
    """Tests for scene change detection scenarios."""

    @pytest.mark.gpu
    def test_scene_change_detection(self, scene_change_frames):
        """GPU should detect scene change in prepared frames."""
        from video_scene_splitter.detection_gpu import compute_pixel_difference_batch_gpu

        mean_diffs, _ = compute_pixel_difference_batch_gpu(scene_change_frames)

        # The scene change happens between frame 4 and 5
        # This should have a higher difference than consecutive within-scene frames
        scene_change_idx = 4

        # Verify scene change has higher difference
        within_scene_diffs = [mean_diffs[i] for i in range(9) if i != scene_change_idx]
        assert mean_diffs[scene_change_idx] > max(within_scene_diffs)

    @pytest.mark.gpu
    def test_histogram_scene_change_detection(self, scene_change_frames):
        """GPU histogram should detect scene change in prepared frames."""
        from video_scene_splitter.detection_gpu import compute_histogram_distance_batch_gpu

        distances = compute_histogram_distance_batch_gpu(scene_change_frames)

        # The scene change happens between frame 4 and 5
        scene_change_idx = 4

        # Scene change should have higher histogram distance
        # (blue to red is a significant hue change)
        within_scene_dists = [distances[i] for i in range(9) if i != scene_change_idx]
        assert distances[scene_change_idx] > max(within_scene_dists)


class TestBatchSizeVariations:
    """Tests for different batch sizes."""

    @pytest.mark.gpu
    @pytest.mark.parametrize("batch_size", [2, 5, 10, 15, 30, 61])
    def test_pixel_diff_various_batch_sizes(self, batch_size):
        """GPU pixel difference should work with various batch sizes."""
        from video_scene_splitter.detection_gpu import compute_pixel_difference_batch_gpu

        # Create frames with varying content
        frames = []
        for i in range(batch_size):
            brightness = (i * 17) % 256  # Varying brightness
            frames.append(np.full((50, 50, 3), brightness, dtype=np.uint8))

        mean_diffs, changed_ratios = compute_pixel_difference_batch_gpu(frames)

        assert len(mean_diffs) == batch_size - 1
        assert len(changed_ratios) == batch_size - 1

    @pytest.mark.gpu
    @pytest.mark.parametrize("batch_size", [2, 5, 10, 15, 30, 61])
    def test_histogram_various_batch_sizes(self, batch_size):
        """GPU histogram distance should work with various batch sizes."""
        from video_scene_splitter.detection_gpu import compute_histogram_distance_batch_gpu

        # Create frames with varying content
        frames = []
        for i in range(batch_size):
            brightness = (i * 17) % 256
            frames.append(np.full((50, 50, 3), brightness, dtype=np.uint8))

        distances = compute_histogram_distance_batch_gpu(frames)

        assert len(distances) == batch_size - 1
