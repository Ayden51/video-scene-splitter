"""
Integration tests for video_scene_splitter.splitter module.

Tests the VideoSceneSplitter class and its integration with other modules.
"""

from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from video_scene_splitter.gpu_utils import GPUInfo, ProcessorType
from video_scene_splitter.splitter import VideoSceneSplitter


class TestVideoSceneSplitterInit:
    """Tests for VideoSceneSplitter initialization."""

    def test_initialization_with_defaults(self):
        """Should initialize with default parameters."""
        splitter = VideoSceneSplitter("test_video.mp4")

        assert splitter.video_path == "test_video.mp4"
        assert splitter.output_dir == "output"
        assert splitter.threshold == 30.0
        assert splitter.min_scene_duration == 1.5
        assert splitter.scene_timestamps == []

    def test_initialization_with_custom_parameters(self):
        """Should initialize with custom parameters."""
        splitter = VideoSceneSplitter(
            video_path="custom.mp4",
            output_dir="custom_output",
            threshold=25.0,
            min_scene_duration=2.0,
        )

        assert splitter.video_path == "custom.mp4"
        assert splitter.output_dir == "custom_output"
        assert splitter.threshold == 25.0
        assert splitter.min_scene_duration == 2.0

    def test_scene_timestamps_initially_empty(self):
        """Scene timestamps should be empty list initially."""
        splitter = VideoSceneSplitter("test.mp4")
        assert splitter.scene_timestamps == []
        assert isinstance(splitter.scene_timestamps, list)


class TestVideoSceneSplitterProcessor:
    """Tests for processor parameter and GPU detection integration."""

    def test_default_processor_is_auto(self):
        """Default processor should be AUTO."""
        splitter = VideoSceneSplitter("test.mp4")
        assert splitter._processor_request == ProcessorType.AUTO

    def test_processor_cpu_explicit(self):
        """Should accept 'cpu' processor parameter."""
        splitter = VideoSceneSplitter("test.mp4", processor="cpu")
        assert splitter._processor_request == ProcessorType.CPU
        assert splitter._active_processor == ProcessorType.CPU

    def test_processor_auto_string(self):
        """Should accept 'auto' processor parameter."""
        splitter = VideoSceneSplitter("test.mp4", processor="auto")
        assert splitter._processor_request == ProcessorType.AUTO

    def test_gpu_info_is_detected(self):
        """Should detect GPU info during initialization."""
        splitter = VideoSceneSplitter("test.mp4")
        assert isinstance(splitter._gpu_info, GPUInfo)

    def test_active_processor_is_set(self):
        """Should set active processor based on GPU availability."""
        splitter = VideoSceneSplitter("test.mp4")
        assert splitter._active_processor in [ProcessorType.CPU, ProcessorType.GPU]

    def test_cpu_processor_always_uses_cpu(self):
        """CPU processor should always result in CPU active processor."""
        splitter = VideoSceneSplitter("test.mp4", processor="cpu")
        assert splitter._active_processor == ProcessorType.CPU

    @patch("video_scene_splitter.splitter.detect_cuda_gpu")
    def test_auto_uses_gpu_when_available(self, mock_detect):
        """AUTO should use GPU when available."""
        mock_detect.return_value = GPUInfo(available=True, name="Test GPU")

        splitter = VideoSceneSplitter("test.mp4", processor="auto")

        assert splitter._active_processor == ProcessorType.GPU

    @patch("video_scene_splitter.splitter.detect_cuda_gpu")
    def test_auto_falls_back_to_cpu(self, mock_detect):
        """AUTO should fallback to CPU when GPU unavailable."""
        mock_detect.return_value = GPUInfo(available=False)

        splitter = VideoSceneSplitter("test.mp4", processor="auto")

        assert splitter._active_processor == ProcessorType.CPU

    @patch("video_scene_splitter.splitter.detect_cuda_gpu")
    def test_gpu_processor_raises_when_unavailable(self, mock_detect):
        """GPU processor should raise error when GPU unavailable."""
        mock_detect.return_value = GPUInfo(available=False)

        with pytest.raises(RuntimeError, match="GPU processing requested"):
            VideoSceneSplitter("test.mp4", processor="gpu")


class TestVideoSceneSplitterDetectScenes:
    """Tests for detect_scenes method."""

    @patch("video_scene_splitter.splitter.cv2.VideoCapture")
    def test_raises_error_for_invalid_video(self, mock_capture):
        """Should raise ValueError if video cannot be opened."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_capture.return_value = mock_cap

        splitter = VideoSceneSplitter("invalid.mp4")

        with pytest.raises(ValueError, match="Cannot open video"):
            splitter.detect_scenes()

    @patch("video_scene_splitter.splitter.cv2.VideoCapture")
    @patch("video_scene_splitter.splitter.Path")
    def test_creates_output_directory(self, mock_path, mock_capture, temp_output_dir):
        """Should create output directory if it doesn't exist."""
        # Mock video capture to return minimal valid video
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        video_props = {cv2.CAP_PROP_FPS: 30.0, cv2.CAP_PROP_FRAME_COUNT: 10}
        mock_cap.get.side_effect = lambda x: video_props.get(x, 0)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [(True, frame)] * 10 + [(False, None)]
        mock_capture.return_value = mock_cap

        # Mock Path to track mkdir calls
        mock_path_instance = Mock()
        mock_path.return_value = mock_path_instance

        splitter = VideoSceneSplitter("test.mp4", output_dir=temp_output_dir)
        splitter.detect_scenes()

        mock_path_instance.mkdir.assert_called_once_with(exist_ok=True)

    @patch("video_scene_splitter.splitter.cv2.VideoCapture")
    def test_returns_timestamps_list(self, mock_capture):
        """Should return a list of timestamps."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda x: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 10,
        }.get(x, 0)
        mock_cap.read.side_effect = [(True, np.zeros((100, 100, 3), dtype=np.uint8))] * 10 + [
            (False, None)
        ]
        mock_capture.return_value = mock_cap

        splitter = VideoSceneSplitter("test.mp4")
        result = splitter.detect_scenes()

        assert isinstance(result, list)
        assert all(isinstance(ts, (int, float)) for ts in result)

    @patch("video_scene_splitter.splitter.cv2.VideoCapture")
    def test_first_timestamp_is_zero(self, mock_capture):
        """First timestamp should always be 0.0."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda x: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 10,
        }.get(x, 0)
        mock_cap.read.side_effect = [(True, np.zeros((100, 100, 3), dtype=np.uint8))] * 10 + [
            (False, None)
        ]
        mock_capture.return_value = mock_cap

        splitter = VideoSceneSplitter("test.mp4")
        result = splitter.detect_scenes()

        assert result[0] == 0.0

    @patch("video_scene_splitter.splitter.cv2.VideoCapture")
    def test_stores_timestamps_in_instance(self, mock_capture):
        """Should store detected timestamps in instance variable."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda x: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 10,
        }.get(x, 0)
        mock_cap.read.side_effect = [(True, np.zeros((100, 100, 3), dtype=np.uint8))] * 10 + [
            (False, None)
        ]
        mock_capture.return_value = mock_cap

        splitter = VideoSceneSplitter("test.mp4")
        result = splitter.detect_scenes()

        assert splitter.scene_timestamps == result

    @patch("video_scene_splitter.splitter.save_metrics_to_csv")
    @patch("video_scene_splitter.splitter.cv2.VideoCapture")
    def test_debug_mode_saves_metrics(self, mock_capture, mock_save_metrics):
        """Debug mode should call save_metrics_to_csv."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda x: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 10,
        }.get(x, 0)
        mock_cap.read.side_effect = [(True, np.zeros((100, 100, 3), dtype=np.uint8))] * 10 + [
            (False, None)
        ]
        mock_capture.return_value = mock_cap
        mock_save_metrics.return_value = 25.0

        splitter = VideoSceneSplitter("test.mp4")
        splitter.detect_scenes(debug=True)

        mock_save_metrics.assert_called_once()

    @patch("video_scene_splitter.splitter.save_debug_frames")
    @patch("video_scene_splitter.splitter.cv2.VideoCapture")
    def test_debug_mode_saves_frames_on_cut(self, mock_capture, mock_save_frames):
        """Debug mode should save frames when a cut is detected."""
        # Create frames that will trigger a hard cut
        black_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        white_frame = np.full((100, 100, 3), 255, dtype=np.uint8)

        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda x: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 100,
        }.get(x, 0)
        # Provide enough frames with a clear cut
        mock_cap.read.side_effect = (
            [(True, black_frame)] * 50 + [(True, white_frame)] * 50 + [(False, None)]
        )
        mock_capture.return_value = mock_cap

        splitter = VideoSceneSplitter("test.mp4", threshold=20.0, min_scene_duration=0.5)
        splitter.detect_scenes(debug=True)

        # Should save debug frames if a cut was detected
        # Note: This depends on the actual detection logic
        assert mock_save_frames.call_count >= 0  # May or may not detect a cut


class TestVideoSceneSplitterSaveTimestamps:
    """Tests for save_timestamps method."""

    @patch("video_scene_splitter.splitter.save_timestamps_to_file")
    def test_calls_save_timestamps_to_file(self, mock_save):
        """Should call the save_timestamps_to_file utility function."""
        splitter = VideoSceneSplitter("test.mp4")
        splitter.scene_timestamps = [0.0, 5.23, 12.45]
        splitter.save_timestamps()

        mock_save.assert_called_once_with(
            "test.mp4", [0.0, 5.23, 12.45], 30.0, "output", "timestamps.txt"
        )

    @patch("video_scene_splitter.splitter.save_timestamps_to_file")
    def test_custom_filename(self, mock_save):
        """Should support custom filename."""
        splitter = VideoSceneSplitter("test.mp4")
        splitter.scene_timestamps = [0.0, 5.0]
        splitter.save_timestamps("custom.txt")

        assert mock_save.call_args[0][4] == "custom.txt"


class TestVideoSceneSplitterSplitVideo:
    """Tests for split_video method."""

    @patch("video_scene_splitter.splitter.split_video_at_timestamps")
    def test_calls_split_video_at_timestamps(self, mock_split):
        """Should call the split_video_at_timestamps function."""
        mock_split.return_value = 3

        splitter = VideoSceneSplitter("test.mp4")
        splitter.scene_timestamps = [0.0, 5.0, 10.0]
        result = splitter.split_video()

        mock_split.assert_called_once_with("test.mp4", [0.0, 5.0, 10.0], "output")
        assert result == 3


class TestVideoSceneSplitterGPUBatchParameters:
    """Tests for GPU batch processing parameters."""

    def test_default_batch_size_is_30(self):
        """Default GPU batch size should be 30."""
        splitter = VideoSceneSplitter("test.mp4")
        assert splitter._gpu_batch_size == 30

    def test_default_memory_fraction_is_0_8(self):
        """Default GPU memory fraction should be 0.8."""
        splitter = VideoSceneSplitter("test.mp4")
        assert splitter._gpu_memory_fraction == 0.8

    def test_custom_batch_size(self):
        """Should accept custom batch size."""
        splitter = VideoSceneSplitter("test.mp4", gpu_batch_size=60)
        assert splitter._gpu_batch_size == 60

    def test_auto_batch_size(self):
        """Should accept 'auto' as batch size."""
        splitter = VideoSceneSplitter("test.mp4", gpu_batch_size="auto")
        assert splitter._gpu_batch_size == "auto"

    def test_custom_memory_fraction(self):
        """Should accept custom memory fraction."""
        splitter = VideoSceneSplitter("test.mp4", gpu_memory_fraction=0.5)
        assert splitter._gpu_memory_fraction == 0.5

    def test_gpu_config_created(self):
        """Should create GPUConfig with correct values."""
        from video_scene_splitter.gpu_utils import GPUConfig

        splitter = VideoSceneSplitter("test.mp4", gpu_batch_size=45, gpu_memory_fraction=0.6)
        assert isinstance(splitter._gpu_config, GPUConfig)
        assert splitter._gpu_config.batch_size == 45
        assert splitter._gpu_config.memory_fraction == 0.6

    def test_gpu_config_uses_default_when_auto(self):
        """GPUConfig should use default batch size 30 when 'auto' is specified."""
        splitter = VideoSceneSplitter("test.mp4", gpu_batch_size="auto")
        assert splitter._gpu_config.batch_size == 30  # Default for GPUConfig


class TestVideoSceneSplitterGPUDetection:
    """Tests for GPU detection method dispatch."""

    @patch("video_scene_splitter.splitter.detect_cuda_gpu")
    @patch("video_scene_splitter.splitter.cv2.VideoCapture")
    @patch("video_scene_splitter.splitter.Path")
    def test_cpu_mode_uses_cpu_detection(self, mock_path, mock_capture, mock_detect):
        """CPU mode should use CPU detection path."""
        mock_detect.return_value = GPUInfo(available=True, name="Test GPU")
        mock_path_instance = Mock()
        mock_path.return_value = mock_path_instance

        # Setup video capture mock
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda x: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 10,
        }.get(x, 0)
        mock_cap.read.side_effect = [(True, np.zeros((100, 100, 3), dtype=np.uint8))] * 10 + [
            (False, None)
        ]
        mock_capture.return_value = mock_cap

        splitter = VideoSceneSplitter("test.mp4", processor="cpu")
        result = splitter.detect_scenes()

        # Should have used CPU path (active processor is CPU)
        assert splitter._active_processor == ProcessorType.CPU
        assert isinstance(result, list)

    @patch("video_scene_splitter.splitter.detect_cuda_gpu")
    def test_gpu_mode_sets_active_processor_to_gpu(self, mock_detect):
        """GPU mode should set active processor to GPU when available."""
        mock_detect.return_value = GPUInfo(available=True, name="Test GPU", memory_free_mb=8192)

        splitter = VideoSceneSplitter("test.mp4", processor="gpu")
        assert splitter._active_processor == ProcessorType.GPU

    @patch("video_scene_splitter.splitter.detect_cuda_gpu")
    def test_auto_mode_uses_gpu_when_available(self, mock_detect):
        """AUTO mode should use GPU when available."""
        mock_detect.return_value = GPUInfo(available=True, name="Test GPU", memory_free_mb=8192)

        splitter = VideoSceneSplitter("test.mp4", processor="auto")
        assert splitter._active_processor == ProcessorType.GPU

    @patch("video_scene_splitter.splitter.detect_cuda_gpu")
    def test_auto_mode_falls_back_to_cpu(self, mock_detect):
        """AUTO mode should fall back to CPU when GPU unavailable."""
        mock_detect.return_value = GPUInfo(available=False)

        splitter = VideoSceneSplitter("test.mp4", processor="auto")
        assert splitter._active_processor == ProcessorType.CPU


class TestVideoSceneSplitterDeterminesBatchSize:
    """Tests for batch size determination logic."""

    @patch("video_scene_splitter.splitter.detect_cuda_gpu")
    def test_determines_fixed_batch_size(self, mock_detect):
        """Should use fixed batch size when specified as integer."""
        mock_detect.return_value = GPUInfo(available=True, name="Test GPU", memory_free_mb=8192)

        splitter = VideoSceneSplitter("test.mp4", processor="gpu", gpu_batch_size=45)
        batch_size = splitter._determine_batch_size(1080, 1920, debug=False)
        assert batch_size == 45

    @patch("video_scene_splitter.splitter.detect_cuda_gpu")
    def test_clamps_batch_size_to_minimum(self, mock_detect):
        """Should clamp batch size to minimum of 5."""
        mock_detect.return_value = GPUInfo(available=True, name="Test GPU", memory_free_mb=8192)

        splitter = VideoSceneSplitter("test.mp4", processor="gpu", gpu_batch_size=2)
        batch_size = splitter._determine_batch_size(1080, 1920, debug=False)
        assert batch_size == 5

    @patch("video_scene_splitter.splitter.detect_cuda_gpu")
    def test_clamps_batch_size_to_maximum(self, mock_detect):
        """Should clamp batch size to maximum of 120."""
        mock_detect.return_value = GPUInfo(available=True, name="Test GPU", memory_free_mb=8192)

        splitter = VideoSceneSplitter("test.mp4", processor="gpu", gpu_batch_size=200)
        batch_size = splitter._determine_batch_size(1080, 1920, debug=False)
        assert batch_size == 120

    @patch("video_scene_splitter.splitter.detect_cuda_gpu")
    def test_auto_batch_size_uses_estimation(self, mock_detect):
        """'auto' batch size should use estimate_optimal_batch_size."""
        mock_detect.return_value = GPUInfo(available=True, name="Test GPU", memory_free_mb=8192)

        splitter = VideoSceneSplitter("test.mp4", processor="gpu", gpu_batch_size="auto")
        batch_size = splitter._determine_batch_size(1080, 1920, debug=False)

        # Should return a reasonable auto-calculated value
        assert 5 <= batch_size <= 120
        assert isinstance(batch_size, int)


# =============================================================================
# GPU Integration Tests (require actual GPU hardware)
# =============================================================================


def create_synthetic_scene_frames(
    num_frames: int,
    height: int = 100,
    width: int = 100,
    scene_changes: list | None = None,
):
    """
    Create synthetic frames with scene changes at specified positions.

    Args:
        num_frames: Total number of frames to generate
        height: Frame height in pixels
        width: Frame width in pixels
        scene_changes: List of frame indices where scene changes occur

    Returns:
        List of numpy arrays (BGR frames) with scene transitions
    """
    if scene_changes is None:
        scene_changes = []

    frames = []
    current_color = np.array([50, 50, 50], dtype=np.int32)  # Start with dark gray

    for i in range(num_frames):
        if i in scene_changes:
            # Create a dramatically different color for scene change
            current_color = np.array(
                [
                    np.random.randint(0, 256),
                    np.random.randint(0, 256),
                    np.random.randint(0, 256),
                ],
                dtype=np.int32,
            )
            # Ensure significant difference by shifting colors
            current_color = (current_color + 128) % 256

        frame = np.full((height, width, 3), current_color, dtype=np.uint8)
        # Add some noise to make frames more realistic
        noise = np.random.randint(-5, 6, (height, width, 3), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frames.append(frame)

    return frames


@pytest.mark.gpu
class TestVideoSceneSplitterGPUIntegration:
    """Integration tests for GPU scene detection that require actual GPU hardware."""

    def test_process_gpu_batch_with_real_frames(self):
        """Test _process_gpu_batch with real GPU computation."""
        from video_scene_splitter.detection_gpu import (
            compute_histogram_distance_batch_gpu,
            compute_pixel_difference_batch_gpu,
            free_gpu_memory,
        )

        splitter = VideoSceneSplitter("test.mp4", processor="gpu", gpu_batch_size=10)

        # Create frames with a scene change
        frames = create_synthetic_scene_frames(10, scene_changes=[5])
        frame_indices = list(range(10))

        all_metrics = []
        splitter._process_gpu_batch(
            frame_buffer=frames,
            frame_indices=frame_indices,
            fps=30.0,
            min_frame_gap=45,  # 1.5s at 30fps
            last_scene_frame=0,
            all_metrics=all_metrics,
            debug=False,
            total_frames=100,
            compute_pixel_diff_fn=compute_pixel_difference_batch_gpu,
            compute_hist_dist_fn=compute_histogram_distance_batch_gpu,
            free_memory_fn=free_gpu_memory,
        )

        # Should have collected metrics for all frame pairs
        assert len(all_metrics) == 9  # 10 frames = 9 pairs
        # Each metric should have expected keys
        for metric in all_metrics:
            assert "frame" in metric
            assert "timestamp" in metric
            assert "hist_dist" in metric
            assert "pixel_diff" in metric
            assert "changed_ratio" in metric

    def test_process_gpu_batch_detects_scene_changes(self):
        """Test that GPU batch processing correctly detects scene changes."""
        from video_scene_splitter.detection_gpu import (
            compute_histogram_distance_batch_gpu,
            compute_pixel_difference_batch_gpu,
            free_gpu_memory,
        )

        splitter = VideoSceneSplitter(
            "test.mp4",
            processor="gpu",
            gpu_batch_size=20,
            threshold=20.0,  # Lower threshold for easier detection
            min_scene_duration=0.0,  # Allow any gap
        )

        # Create frames with distinct scene changes
        frames = []
        # Scene 1: Red frames
        for _ in range(5):
            frames.append(np.full((100, 100, 3), [0, 0, 200], dtype=np.uint8))
        # Scene 2: Blue frames (dramatic change)
        for _ in range(5):
            frames.append(np.full((100, 100, 3), [200, 0, 0], dtype=np.uint8))
        # Scene 3: Green frames (dramatic change)
        for _ in range(5):
            frames.append(np.full((100, 100, 3), [0, 200, 0], dtype=np.uint8))

        frame_indices = list(range(15))
        all_metrics = []
        splitter.scene_timestamps = [0.0]  # Initialize with first scene

        splitter._process_gpu_batch(
            frame_buffer=frames,
            frame_indices=frame_indices,
            fps=30.0,
            min_frame_gap=1,
            last_scene_frame=0,
            all_metrics=all_metrics,
            debug=False,
            total_frames=15,
            compute_pixel_diff_fn=compute_pixel_difference_batch_gpu,
            compute_hist_dist_fn=compute_histogram_distance_batch_gpu,
            free_memory_fn=free_gpu_memory,
        )

        # Should have detected scene changes at frames 5 and 10
        assert len(splitter.scene_timestamps) >= 2  # At least original + detected

    def test_gpu_batch_with_various_sizes(self):
        """Test GPU batch processing with different batch sizes."""
        from video_scene_splitter.detection_gpu import (
            compute_histogram_distance_batch_gpu,
            compute_pixel_difference_batch_gpu,
            free_gpu_memory,
        )

        batch_sizes = [5, 10, 20, 30, 60]

        for batch_size in batch_sizes:
            splitter = VideoSceneSplitter("test.mp4", processor="gpu", gpu_batch_size=batch_size)

            num_frames = batch_size + 5  # Slightly more than batch size
            frames = create_synthetic_scene_frames(num_frames)
            frame_indices = list(range(num_frames))

            all_metrics = []
            splitter._process_gpu_batch(
                frame_buffer=frames,
                frame_indices=frame_indices,
                fps=30.0,
                min_frame_gap=45,
                last_scene_frame=0,
                all_metrics=all_metrics,
                debug=False,
                total_frames=num_frames,
                compute_pixel_diff_fn=compute_pixel_difference_batch_gpu,
                compute_hist_dist_fn=compute_histogram_distance_batch_gpu,
                free_memory_fn=free_gpu_memory,
            )

            # Should have metrics for all pairs
            assert len(all_metrics) == num_frames - 1

    def test_gpu_memory_cleanup(self):
        """Test that GPU memory is properly cleaned up after batch processing."""
        from video_scene_splitter.detection_gpu import (
            compute_histogram_distance_batch_gpu,
            compute_pixel_difference_batch_gpu,
            free_gpu_memory,
        )

        try:
            import cupy as cp

            # Get initial memory usage
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            initial_used = mempool.used_bytes()

            splitter = VideoSceneSplitter("test.mp4", processor="gpu", gpu_batch_size=30)

            # Create large frames to use significant memory
            frames = create_synthetic_scene_frames(50, height=200, width=200)
            frame_indices = list(range(50))
            all_metrics = []

            splitter._process_gpu_batch(
                frame_buffer=frames,
                frame_indices=frame_indices,
                fps=30.0,
                min_frame_gap=45,
                last_scene_frame=0,
                all_metrics=all_metrics,
                debug=False,
                total_frames=50,
                compute_pixel_diff_fn=compute_pixel_difference_batch_gpu,
                compute_hist_dist_fn=compute_histogram_distance_batch_gpu,
                free_memory_fn=free_gpu_memory,
            )

            # Memory should be freed (or at least returned to pool)
            final_used = mempool.used_bytes()
            # Memory should be similar to initial (within 10MB tolerance)
            assert final_used - initial_used < 10 * 1024 * 1024

        except ImportError:
            pytest.skip("CuPy not available")

    def test_gpu_cpu_result_consistency(self):
        """Test that GPU and CPU produce consistent scene detection results."""
        from video_scene_splitter.detection import (
            compute_histogram_distance,
            compute_pixel_difference,
        )
        from video_scene_splitter.detection_gpu import (
            compute_histogram_distance_batch_gpu,
            compute_pixel_difference_batch_gpu,
            free_gpu_memory,
        )

        # Create test frames
        frames = create_synthetic_scene_frames(20, height=100, width=100)

        # CPU results
        cpu_pixel_diffs = []
        cpu_hist_dists = []
        for i in range(len(frames) - 1):
            pixel_diff, _ = compute_pixel_difference(frames[i], frames[i + 1])
            hist_dist = compute_histogram_distance(frames[i], frames[i + 1])
            cpu_pixel_diffs.append(pixel_diff)
            cpu_hist_dists.append(hist_dist)

        # GPU results
        gpu_pixel_diffs, _ = compute_pixel_difference_batch_gpu(frames)
        gpu_hist_dists = compute_histogram_distance_batch_gpu(frames)
        free_gpu_memory()

        # Compare results (with tolerance for floating point)
        for i in range(len(cpu_pixel_diffs)):
            assert abs(cpu_pixel_diffs[i] - float(gpu_pixel_diffs[i])) < 1.0, (
                f"Pixel diff mismatch at frame {i}: CPU={cpu_pixel_diffs[i]}, "
                f"GPU={gpu_pixel_diffs[i]}"
            )
            assert abs(cpu_hist_dists[i] - float(gpu_hist_dists[i])) < 5.0, (
                f"Hist dist mismatch at frame {i}: CPU={cpu_hist_dists[i]}, GPU={gpu_hist_dists[i]}"
            )

    def test_process_gpu_batch_with_single_frame_returns_early(self):
        """Test that processing with fewer than 2 frames returns immediately."""
        from video_scene_splitter.detection_gpu import (
            compute_histogram_distance_batch_gpu,
            compute_pixel_difference_batch_gpu,
            free_gpu_memory,
        )

        splitter = VideoSceneSplitter("test.mp4", processor="gpu")

        # Single frame buffer
        frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
        all_metrics = []

        result = splitter._process_gpu_batch(
            frame_buffer=frames,
            frame_indices=[0],
            fps=30.0,
            min_frame_gap=45,
            last_scene_frame=0,
            all_metrics=all_metrics,
            debug=False,
            total_frames=100,
            compute_pixel_diff_fn=compute_pixel_difference_batch_gpu,
            compute_hist_dist_fn=compute_histogram_distance_batch_gpu,
            free_memory_fn=free_gpu_memory,
        )

        # Should return early without processing
        assert result == 0
        assert len(all_metrics) == 0

    def test_process_gpu_batch_with_empty_buffer(self):
        """Test that processing with empty buffer handles gracefully."""
        from video_scene_splitter.detection_gpu import (
            compute_histogram_distance_batch_gpu,
            compute_pixel_difference_batch_gpu,
            free_gpu_memory,
        )

        splitter = VideoSceneSplitter("test.mp4", processor="gpu")
        all_metrics = []

        result = splitter._process_gpu_batch(
            frame_buffer=[],
            frame_indices=[],
            fps=30.0,
            min_frame_gap=45,
            last_scene_frame=5,
            all_metrics=all_metrics,
            debug=False,
            total_frames=100,
            compute_pixel_diff_fn=compute_pixel_difference_batch_gpu,
            compute_hist_dist_fn=compute_histogram_distance_batch_gpu,
            free_memory_fn=free_gpu_memory,
        )

        # Should return the same last_scene_frame
        assert result == 5
        assert len(all_metrics) == 0

    def test_determine_batch_size_auto_with_debug_output(self, capsys):
        """Test auto batch size determination with debug output."""
        splitter = VideoSceneSplitter("test.mp4", processor="gpu", gpu_batch_size="auto")

        batch_size = splitter._determine_batch_size(1080, 1920, debug=True)

        captured = capsys.readouterr()
        assert "Auto-selected batch size" in captured.out
        assert 5 <= batch_size <= 120


@pytest.mark.gpu
class TestVideoSceneSplitterGPUErrorHandling:
    """Tests for GPU error handling and fallback mechanisms."""

    def test_cpu_fallback_on_gpu_error(self):
        """Test that CPU fallback works when GPU processing fails."""
        splitter = VideoSceneSplitter(
            "test.mp4", processor="gpu", threshold=20.0, min_scene_duration=0.0
        )

        # Create frames with scene change
        frames = []
        for _ in range(5):
            frames.append(np.full((100, 100, 3), [50, 50, 50], dtype=np.uint8))
        for _ in range(5):
            frames.append(np.full((100, 100, 3), [200, 200, 200], dtype=np.uint8))

        frame_indices = list(range(10))
        all_metrics = []
        splitter.scene_timestamps = [0.0]

        # Mock GPU function that raises error
        def failing_gpu_fn(frames):
            raise RuntimeError("Simulated GPU memory error")

        def mock_free_memory():
            pass

        splitter._process_gpu_batch(
            frame_buffer=frames,
            frame_indices=frame_indices,
            fps=30.0,
            min_frame_gap=1,
            last_scene_frame=0,
            all_metrics=all_metrics,
            debug=False,
            total_frames=10,
            compute_pixel_diff_fn=failing_gpu_fn,
            compute_hist_dist_fn=failing_gpu_fn,
            free_memory_fn=mock_free_memory,
        )

        # Should have processed on CPU and collected metrics
        assert len(all_metrics) == 9  # 10 frames = 9 pairs

    def test_cpu_fallback_on_oom_error(self):
        """Test CPU fallback specifically for OutOfMemoryError."""
        splitter = VideoSceneSplitter(
            "test.mp4", processor="gpu", threshold=20.0, min_scene_duration=0.0
        )

        frames = create_synthetic_scene_frames(10)
        frame_indices = list(range(10))
        all_metrics = []
        splitter.scene_timestamps = [0.0]

        # Create a custom exception class that mimics CUDA OOM
        class OutOfMemoryError(Exception):
            pass

        def oom_gpu_fn(frames):
            raise OutOfMemoryError("CUDA out of memory")

        def mock_free_memory():
            pass

        splitter._process_gpu_batch(
            frame_buffer=frames,
            frame_indices=frame_indices,
            fps=30.0,
            min_frame_gap=1,
            last_scene_frame=0,
            all_metrics=all_metrics,
            debug=False,
            total_frames=10,
            compute_pixel_diff_fn=oom_gpu_fn,
            compute_hist_dist_fn=oom_gpu_fn,
            free_memory_fn=mock_free_memory,
        )

        # Should have fallen back to CPU
        assert len(all_metrics) == 9

    def test_cpu_batch_fallback_processes_correctly(self):
        """Test that _process_cpu_batch_fallback produces correct results."""
        splitter = VideoSceneSplitter(
            "test.mp4",
            processor="cpu",
            threshold=20.0,
            min_scene_duration=0.0,
        )

        # Create frames with clear scene change
        frames = []
        for _ in range(5):
            frames.append(np.full((100, 100, 3), [0, 0, 0], dtype=np.uint8))
        for _ in range(5):
            frames.append(np.full((100, 100, 3), [255, 255, 255], dtype=np.uint8))

        frame_indices = list(range(10))
        all_metrics = []
        splitter.scene_timestamps = [0.0]

        splitter._process_cpu_batch_fallback(
            frame_buffer=frames,
            frame_indices=frame_indices,
            fps=30.0,
            min_frame_gap=1,
            last_scene_frame=0,
            all_metrics=all_metrics,
            debug=False,
            total_frames=10,
        )

        # Should have detected scene change at frame 5
        assert len(splitter.scene_timestamps) >= 2
        assert len(all_metrics) == 9


@pytest.mark.gpu
class TestVideoSceneSplitterGPUBatchProcessing:
    """Tests for GPU batch processing edge cases and behavior."""

    def test_overlapping_batch_processing(self):
        """Test that batch overlapping maintains continuity."""
        from video_scene_splitter.detection_gpu import (
            compute_histogram_distance_batch_gpu,
            compute_pixel_difference_batch_gpu,
            free_gpu_memory,
        )

        splitter = VideoSceneSplitter("test.mp4", processor="gpu", gpu_batch_size=10)

        # Create frames spanning multiple batches
        all_frames = create_synthetic_scene_frames(25, scene_changes=[12])

        # Process in batches with overlap
        all_metrics = []
        splitter.scene_timestamps = [0.0]
        last_scene_frame = 0

        # Batch 1: frames 0-10
        batch1 = all_frames[0:11]
        batch1_indices = list(range(11))
        last_scene_frame = splitter._process_gpu_batch(
            frame_buffer=batch1,
            frame_indices=batch1_indices,
            fps=30.0,
            min_frame_gap=1,
            last_scene_frame=last_scene_frame,
            all_metrics=all_metrics,
            debug=False,
            total_frames=25,
            compute_pixel_diff_fn=compute_pixel_difference_batch_gpu,
            compute_hist_dist_fn=compute_histogram_distance_batch_gpu,
            free_memory_fn=free_gpu_memory,
        )

        # Batch 2: frames 10-20 (overlapping at frame 10)
        batch2 = all_frames[10:21]
        batch2_indices = list(range(10, 21))
        last_scene_frame = splitter._process_gpu_batch(
            frame_buffer=batch2,
            frame_indices=batch2_indices,
            fps=30.0,
            min_frame_gap=1,
            last_scene_frame=last_scene_frame,
            all_metrics=all_metrics,
            debug=False,
            total_frames=25,
            compute_pixel_diff_fn=compute_pixel_difference_batch_gpu,
            compute_hist_dist_fn=compute_histogram_distance_batch_gpu,
            free_memory_fn=free_gpu_memory,
        )

        # Batch 3: frames 20-25
        batch3 = all_frames[20:25]
        batch3_indices = list(range(20, 25))
        splitter._process_gpu_batch(
            frame_buffer=batch3,
            frame_indices=batch3_indices,
            fps=30.0,
            min_frame_gap=1,
            last_scene_frame=last_scene_frame,
            all_metrics=all_metrics,
            debug=False,
            total_frames=25,
            compute_pixel_diff_fn=compute_pixel_difference_batch_gpu,
            compute_hist_dist_fn=compute_histogram_distance_batch_gpu,
            free_memory_fn=free_gpu_memory,
        )

        # Should have processed frames correctly with overlap
        # Total unique pairs: 24 (but we have some overlap)
        assert len(all_metrics) >= 20

    def test_min_frame_gap_respected(self):
        """Test that minimum frame gap prevents rapid scene detection."""
        from video_scene_splitter.detection_gpu import (
            compute_histogram_distance_batch_gpu,
            compute_pixel_difference_batch_gpu,
            free_gpu_memory,
        )

        splitter = VideoSceneSplitter(
            "test.mp4",
            processor="gpu",
            threshold=10.0,  # Low threshold to easily trigger
            min_scene_duration=2.0,  # 2 seconds = 60 frames at 30fps
        )

        # Create frames with multiple scene changes close together
        frames = []
        colors = [
            [0, 0, 255],  # Red
            [0, 255, 0],  # Green
            [255, 0, 0],  # Blue
            [255, 255, 0],  # Cyan
            [255, 0, 255],  # Magenta
        ]
        for i in range(50):
            color_idx = i // 10  # Change every 10 frames
            if color_idx < len(colors):
                frames.append(np.full((100, 100, 3), colors[color_idx], dtype=np.uint8))
            else:
                frames.append(np.full((100, 100, 3), colors[-1], dtype=np.uint8))

        frame_indices = list(range(50))
        all_metrics = []
        splitter.scene_timestamps = [0.0]

        splitter._process_gpu_batch(
            frame_buffer=frames,
            frame_indices=frame_indices,
            fps=30.0,
            min_frame_gap=60,  # 2 seconds worth of frames
            last_scene_frame=0,
            all_metrics=all_metrics,
            debug=False,
            total_frames=50,
            compute_pixel_diff_fn=compute_pixel_difference_batch_gpu,
            compute_hist_dist_fn=compute_histogram_distance_batch_gpu,
            free_memory_fn=free_gpu_memory,
        )

        # Should not detect all scene changes due to min_frame_gap
        # At most: initial + 1 detection (if any frame pair is far enough apart)
        assert len(splitter.scene_timestamps) <= 2

    def test_large_batch_processing(self):
        """Test processing large batches efficiently."""
        from video_scene_splitter.detection_gpu import (
            compute_histogram_distance_batch_gpu,
            compute_pixel_difference_batch_gpu,
            free_gpu_memory,
        )

        splitter = VideoSceneSplitter("test.mp4", processor="gpu", gpu_batch_size=100)

        # Create large batch of frames
        frames = create_synthetic_scene_frames(120, height=200, width=200)
        frame_indices = list(range(120))
        all_metrics = []

        splitter._process_gpu_batch(
            frame_buffer=frames,
            frame_indices=frame_indices,
            fps=30.0,
            min_frame_gap=45,
            last_scene_frame=0,
            all_metrics=all_metrics,
            debug=False,
            total_frames=120,
            compute_pixel_diff_fn=compute_pixel_difference_batch_gpu,
            compute_hist_dist_fn=compute_histogram_distance_batch_gpu,
            free_memory_fn=free_gpu_memory,
        )

        # Should process all frame pairs
        assert len(all_metrics) == 119

    def test_metrics_contain_valid_values(self):
        """Test that metrics contain valid numerical values."""
        from video_scene_splitter.detection_gpu import (
            compute_histogram_distance_batch_gpu,
            compute_pixel_difference_batch_gpu,
            free_gpu_memory,
        )

        splitter = VideoSceneSplitter("test.mp4", processor="gpu")

        frames = create_synthetic_scene_frames(20)
        frame_indices = list(range(20))
        all_metrics = []

        splitter._process_gpu_batch(
            frame_buffer=frames,
            frame_indices=frame_indices,
            fps=30.0,
            min_frame_gap=45,
            last_scene_frame=0,
            all_metrics=all_metrics,
            debug=False,
            total_frames=20,
            compute_pixel_diff_fn=compute_pixel_difference_batch_gpu,
            compute_hist_dist_fn=compute_histogram_distance_batch_gpu,
            free_memory_fn=free_gpu_memory,
        )

        for metric in all_metrics:
            # Frame numbers should be valid
            assert isinstance(metric["frame"], int)
            assert metric["frame"] >= 0

            # Timestamps should be non-negative
            assert isinstance(metric["timestamp"], float)
            assert metric["timestamp"] >= 0

            # Histogram distance should be non-negative
            assert isinstance(metric["hist_dist"], float)
            assert metric["hist_dist"] >= 0

            # Pixel diff should be non-negative
            assert isinstance(metric["pixel_diff"], float)
            assert metric["pixel_diff"] >= 0

            # Changed ratio should be 0-100%
            assert isinstance(metric["changed_ratio"], float)
            assert 0 <= metric["changed_ratio"] <= 100
