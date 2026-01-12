"""
Unit tests for video_scene_splitter.video_processor module.

Tests the video splitting functionality with mocked FFmpeg calls,
NVENC encoder detection, and async frame reading.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from video_scene_splitter.video_processor import (
    AsyncFrameReader,
    _get_libx264_options,
    _get_nvenc_options,
    detect_nvenc_support,
    get_encoder_info,
    get_encoder_options,
    read_frames_async,
    split_video_at_timestamps,
)


class TestSplitVideoAtTimestamps:
    """Tests for split_video_at_timestamps function."""

    @patch("video_scene_splitter.video_processor.subprocess.run")
    def test_returns_zero_for_empty_timestamps(self, mock_run, temp_output_dir):
        """Should return 0 when no timestamps are provided."""
        result = split_video_at_timestamps("video.mp4", [], temp_output_dir)
        assert result == 0
        mock_run.assert_not_called()

    @patch("video_scene_splitter.video_processor.subprocess.run")
    @patch("video_scene_splitter.video_processor.detect_nvenc_support")
    def test_creates_correct_number_of_scenes(self, mock_nvenc, mock_run, temp_output_dir):
        """Should create one file per scene."""
        mock_nvenc.return_value = False  # Use libx264
        mock_run.return_value = MagicMock(returncode=0)
        timestamps = [0.0, 5.23, 12.45]

        result = split_video_at_timestamps("video.mp4", timestamps, temp_output_dir)

        assert result == 3
        assert mock_run.call_count == 3

    @patch("video_scene_splitter.video_processor.subprocess.run")
    @patch("video_scene_splitter.video_processor.detect_nvenc_support")
    def test_output_filenames_are_correct(self, mock_nvenc, mock_run, temp_output_dir):
        """Output files should have correct naming pattern."""
        mock_nvenc.return_value = False
        mock_run.return_value = MagicMock(returncode=0)
        timestamps = [0.0, 5.23]

        split_video_at_timestamps("my_video.mp4", timestamps, temp_output_dir)

        calls = mock_run.call_args_list
        # Check that the output filename (last argument) contains the expected pattern
        assert "my_video_scene_001.mp4" in calls[0][0][0][-1]
        assert "my_video_scene_002.mp4" in calls[1][0][0][-1]

    @patch("video_scene_splitter.video_processor.subprocess.run")
    @patch("video_scene_splitter.video_processor.detect_nvenc_support")
    def test_preserves_video_extension(self, mock_nvenc, mock_run, temp_output_dir):
        """Should preserve the original video file extension."""
        mock_nvenc.return_value = False
        mock_run.return_value = MagicMock(returncode=0)
        timestamps = [0.0, 5.0]

        split_video_at_timestamps("video.mkv", timestamps, temp_output_dir)

        calls = mock_run.call_args_list
        # Check that the output filename (last argument) contains the expected pattern
        assert "video_scene_001.mkv" in calls[0][0][0][-1]

    @patch("video_scene_splitter.video_processor.subprocess.run")
    @patch("video_scene_splitter.video_processor.detect_nvenc_support")
    def test_ffmpeg_command_structure(self, mock_nvenc, mock_run, temp_output_dir):
        """FFmpeg command should have correct structure."""
        mock_nvenc.return_value = False
        mock_run.return_value = MagicMock(returncode=0)
        timestamps = [0.0, 5.23]

        split_video_at_timestamps("video.mp4", timestamps, temp_output_dir)

        cmd = mock_run.call_args_list[0][0][0]
        assert cmd[0] == "ffmpeg"
        assert "-i" in cmd
        assert "video.mp4" in cmd
        assert "-ss" in cmd
        assert "-c:v" in cmd
        assert "libx264" in cmd
        assert "-c:a" in cmd
        assert "aac" in cmd

    @patch("video_scene_splitter.video_processor.subprocess.run")
    @patch("video_scene_splitter.video_processor.detect_nvenc_support")
    def test_first_scene_start_time(self, mock_nvenc, mock_run, temp_output_dir):
        """First scene should start at the first timestamp."""
        mock_nvenc.return_value = False
        mock_run.return_value = MagicMock(returncode=0)
        timestamps = [0.0, 5.23, 12.45]

        split_video_at_timestamps("video.mp4", timestamps, temp_output_dir)

        cmd = mock_run.call_args_list[0][0][0]
        ss_index = cmd.index("-ss")
        assert cmd[ss_index + 1] == "0.0"

    @patch("video_scene_splitter.video_processor.subprocess.run")
    @patch("video_scene_splitter.video_processor.detect_nvenc_support")
    def test_middle_scene_has_end_time(self, mock_nvenc, mock_run, temp_output_dir):
        """Middle scenes should have both start and end times."""
        mock_nvenc.return_value = False
        mock_run.return_value = MagicMock(returncode=0)
        timestamps = [0.0, 5.23, 12.45]

        split_video_at_timestamps("video.mp4", timestamps, temp_output_dir)

        # Check second scene (index 1)
        cmd = mock_run.call_args_list[1][0][0]
        assert "-ss" in cmd
        assert "-to" in cmd
        ss_index = cmd.index("-ss")
        to_index = cmd.index("-to")
        assert cmd[ss_index + 1] == "5.23"
        assert cmd[to_index + 1] == "12.45"

    @patch("video_scene_splitter.video_processor.subprocess.run")
    @patch("video_scene_splitter.video_processor.detect_nvenc_support")
    def test_last_scene_no_end_time(self, mock_nvenc, mock_run, temp_output_dir):
        """Last scene should not have -to flag (goes to end of video)."""
        mock_nvenc.return_value = False
        mock_run.return_value = MagicMock(returncode=0)
        timestamps = [0.0, 5.23, 12.45]

        split_video_at_timestamps("video.mp4", timestamps, temp_output_dir)

        # Check last scene (index 2)
        cmd = mock_run.call_args_list[2][0][0]
        assert "-to" not in cmd
        ss_index = cmd.index("-ss")
        assert cmd[ss_index + 1] == "12.45"

    @patch("video_scene_splitter.video_processor.subprocess.run")
    @patch("video_scene_splitter.video_processor.detect_nvenc_support")
    def test_handles_ffmpeg_failure(self, mock_nvenc, mock_run, temp_output_dir):
        """Should handle FFmpeg failures gracefully."""
        mock_nvenc.return_value = False
        # First call succeeds, second fails, third succeeds
        mock_run.side_effect = [
            MagicMock(returncode=0),
            MagicMock(returncode=1),
            MagicMock(returncode=0),
        ]
        timestamps = [0.0, 5.0, 10.0]

        result = split_video_at_timestamps("video.mp4", timestamps, temp_output_dir)

        assert result == 2  # Only 2 successful
        assert mock_run.call_count == 3

    @patch("video_scene_splitter.video_processor.subprocess.run")
    @patch("video_scene_splitter.video_processor.detect_nvenc_support")
    def test_single_scene(self, mock_nvenc, mock_run, temp_output_dir):
        """Should handle single scene correctly."""
        mock_nvenc.return_value = False
        mock_run.return_value = MagicMock(returncode=0)
        timestamps = [0.0]

        result = split_video_at_timestamps("video.mp4", timestamps, temp_output_dir)

        assert result == 1
        assert mock_run.call_count == 1

    @patch("video_scene_splitter.video_processor.subprocess.run")
    @patch("video_scene_splitter.video_processor.detect_nvenc_support")
    def test_output_directory_in_path(self, mock_nvenc, mock_run, temp_output_dir):
        """Output files should be placed in the specified directory."""
        mock_nvenc.return_value = False
        mock_run.return_value = MagicMock(returncode=0)
        timestamps = [0.0, 5.0]

        split_video_at_timestamps("video.mp4", timestamps, temp_output_dir)

        cmd = mock_run.call_args_list[0][0][0]
        output_file = cmd[-1]
        assert temp_output_dir in output_file

    @patch("video_scene_splitter.video_processor.subprocess.run")
    @patch("video_scene_splitter.video_processor.detect_nvenc_support")
    def test_overwrite_flag_present(self, mock_nvenc, mock_run, temp_output_dir):
        """FFmpeg command should include -y flag to overwrite files."""
        mock_nvenc.return_value = False
        mock_run.return_value = MagicMock(returncode=0)
        timestamps = [0.0]

        split_video_at_timestamps("video.mp4", timestamps, temp_output_dir)

        cmd = mock_run.call_args_list[0][0][0]
        assert "-y" in cmd

    @patch("video_scene_splitter.video_processor.subprocess.run")
    @patch("video_scene_splitter.video_processor.detect_nvenc_support")
    def test_capture_output_enabled(self, mock_nvenc, mock_run, temp_output_dir):
        """subprocess.run should be called with capture_output=True."""
        mock_nvenc.return_value = False
        mock_run.return_value = MagicMock(returncode=0)
        timestamps = [0.0]

        split_video_at_timestamps("video.mp4", timestamps, temp_output_dir)

        assert mock_run.call_args_list[0][1]["capture_output"] is True

    @patch("video_scene_splitter.video_processor.subprocess.run")
    @patch("video_scene_splitter.video_processor.detect_nvenc_support")
    def test_uses_nvenc_when_available_auto_mode(self, mock_nvenc, mock_run, temp_output_dir):
        """Should use NVENC encoder when available in auto mode."""
        mock_nvenc.return_value = True
        mock_run.return_value = MagicMock(returncode=0)
        timestamps = [0.0]

        split_video_at_timestamps("video.mp4", timestamps, temp_output_dir, processor="auto")

        cmd = mock_run.call_args_list[0][0][0]
        assert "h264_nvenc" in cmd

    @patch("video_scene_splitter.video_processor.subprocess.run")
    @patch("video_scene_splitter.video_processor.detect_nvenc_support")
    def test_uses_libx264_when_nvenc_unavailable_auto_mode(
        self, mock_nvenc, mock_run, temp_output_dir
    ):
        """Should fall back to libx264 when NVENC unavailable in auto mode."""
        mock_nvenc.return_value = False
        mock_run.return_value = MagicMock(returncode=0)
        timestamps = [0.0]

        split_video_at_timestamps("video.mp4", timestamps, temp_output_dir, processor="auto")

        cmd = mock_run.call_args_list[0][0][0]
        assert "libx264" in cmd
        assert "h264_nvenc" not in cmd

    @patch("video_scene_splitter.video_processor.subprocess.run")
    @patch("video_scene_splitter.video_processor.detect_nvenc_support")
    def test_uses_libx264_in_cpu_mode(self, mock_nvenc, mock_run, temp_output_dir):
        """Should always use libx264 in CPU mode."""
        mock_nvenc.return_value = True  # Even if NVENC is available
        mock_run.return_value = MagicMock(returncode=0)
        timestamps = [0.0]

        split_video_at_timestamps("video.mp4", timestamps, temp_output_dir, processor="cpu")

        cmd = mock_run.call_args_list[0][0][0]
        assert "libx264" in cmd
        assert "h264_nvenc" not in cmd


class TestNVENCDetection:
    """Tests for NVENC encoder detection."""

    @patch("video_scene_splitter.video_processor.subprocess.run")
    def test_detect_nvenc_support_available(self, mock_run):
        """Should return True when h264_nvenc is in FFmpeg encoders."""
        # Reset cached value
        import video_scene_splitter.video_processor as vp

        vp._nvenc_available = None

        mock_run.return_value = MagicMock(
            stdout="V..... h264_nvenc           NVIDIA NVENC H.264 encoder",
            returncode=0,
        )

        result = detect_nvenc_support()
        assert result is True

    @patch("video_scene_splitter.video_processor.subprocess.run")
    def test_detect_nvenc_support_unavailable(self, mock_run):
        """Should return False when h264_nvenc is not in FFmpeg encoders."""
        import video_scene_splitter.video_processor as vp

        vp._nvenc_available = None

        mock_run.return_value = MagicMock(
            stdout="V..... libx264              libx264 H.264 encoder",
            returncode=0,
        )

        result = detect_nvenc_support()
        assert result is False

    @patch("video_scene_splitter.video_processor.subprocess.run")
    def test_detect_nvenc_support_caches_result(self, mock_run):
        """Should cache the detection result."""
        import video_scene_splitter.video_processor as vp

        vp._nvenc_available = None

        mock_run.return_value = MagicMock(
            stdout="V..... h264_nvenc           NVIDIA NVENC H.264 encoder",
            returncode=0,
        )

        # First call
        result1 = detect_nvenc_support()
        # Second call should use cached value
        result2 = detect_nvenc_support()

        assert result1 is True
        assert result2 is True
        # Should only call subprocess once due to caching
        assert mock_run.call_count == 1

    @patch("video_scene_splitter.video_processor.subprocess.run")
    def test_detect_nvenc_support_handles_timeout(self, mock_run):
        """Should return False on timeout."""
        import subprocess

        import video_scene_splitter.video_processor as vp

        vp._nvenc_available = None

        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ffmpeg", timeout=10)

        result = detect_nvenc_support()
        assert result is False


class TestEncoderOptions:
    """Tests for encoder option functions."""

    def test_get_libx264_options(self):
        """Should return correct libx264 options."""
        options = _get_libx264_options()
        assert options == ["-c:v", "libx264"]

    def test_get_nvenc_options_default_preset(self):
        """Should return NVENC options with default preset."""
        options = _get_nvenc_options()
        assert "-c:v" in options
        assert "h264_nvenc" in options
        assert "-preset" in options
        assert "p4" in options  # Default preset

    def test_get_nvenc_options_custom_preset(self):
        """Should return NVENC options with custom preset."""
        options = _get_nvenc_options(preset="p7")
        assert "p7" in options

    @patch("video_scene_splitter.video_processor.detect_nvenc_support")
    def test_get_encoder_options_cpu_mode(self, mock_nvenc):
        """CPU mode should always return libx264 options."""
        mock_nvenc.return_value = True  # Even if NVENC available

        options = get_encoder_options("cpu")
        assert options == ["-c:v", "libx264"]

    @patch("video_scene_splitter.video_processor.detect_nvenc_support")
    def test_get_encoder_options_gpu_mode_available(self, mock_nvenc):
        """GPU mode should return NVENC options when available."""
        mock_nvenc.return_value = True

        options = get_encoder_options("gpu")
        assert "h264_nvenc" in options

    @patch("video_scene_splitter.video_processor.detect_nvenc_support")
    def test_get_encoder_options_gpu_mode_unavailable(self, mock_nvenc):
        """GPU mode should raise error when NVENC unavailable."""
        mock_nvenc.return_value = False

        with pytest.raises(RuntimeError, match="NVENC is not available"):
            get_encoder_options("gpu")

    @patch("video_scene_splitter.video_processor.detect_nvenc_support")
    def test_get_encoder_options_auto_mode_nvenc_available(self, mock_nvenc):
        """Auto mode should use NVENC when available."""
        mock_nvenc.return_value = True

        options = get_encoder_options("auto")
        assert "h264_nvenc" in options

    @patch("video_scene_splitter.video_processor.detect_nvenc_support")
    def test_get_encoder_options_auto_mode_nvenc_unavailable(self, mock_nvenc):
        """Auto mode should fall back to libx264 when NVENC unavailable."""
        mock_nvenc.return_value = False

        options = get_encoder_options("auto")
        assert options == ["-c:v", "libx264"]


class TestEncoderInfo:
    """Tests for get_encoder_info function."""

    @patch("video_scene_splitter.video_processor.detect_nvenc_support")
    def test_encoder_info_cpu_mode(self, mock_nvenc):
        """CPU mode should report libx264 software encoder."""
        mock_nvenc.return_value = True

        info = get_encoder_info("cpu")
        assert info["name"] == "libx264"
        assert info["type"] == "software"
        assert info["selected_processor"] == "cpu"

    @patch("video_scene_splitter.video_processor.detect_nvenc_support")
    def test_encoder_info_gpu_mode_available(self, mock_nvenc):
        """GPU mode should report NVENC hardware encoder when available."""
        mock_nvenc.return_value = True

        info = get_encoder_info("gpu")
        assert info["name"] == "h264_nvenc"
        assert info["type"] == "hardware"
        assert info["nvenc_available"] is True

    @patch("video_scene_splitter.video_processor.detect_nvenc_support")
    def test_encoder_info_auto_mode(self, mock_nvenc):
        """Auto mode should report appropriate encoder based on availability."""
        mock_nvenc.return_value = True

        info = get_encoder_info("auto")
        assert info["name"] == "h264_nvenc"
        assert info["type"] == "hardware"
        assert info["selected_processor"] == "auto"


class TestAsyncFrameReader:
    """Tests for AsyncFrameReader class and read_frames_async function."""

    def test_async_frame_reader_iterates_batches(self):
        """AsyncFrameReader should yield batches of frames."""
        # Create a mock video capture
        mock_cap = MagicMock()
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(10)]
        frame_index = [0]

        def mock_read():
            if frame_index[0] < len(frames):
                frame = frames[frame_index[0]]
                frame_index[0] += 1
                return True, frame
            return False, None

        mock_cap.read = mock_read

        reader = AsyncFrameReader(mock_cap, batch_size=3)
        batches = list(reader)
        reader.close()

        # Should have 4 batches: 3, 3, 3, 1
        assert len(batches) == 4
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3
        assert len(batches[2]) == 3
        assert len(batches[3]) == 1

    def test_async_frame_reader_context_manager(self):
        """AsyncFrameReader should work as context manager."""
        mock_cap = MagicMock()
        mock_cap.read.return_value = (False, None)

        with AsyncFrameReader(mock_cap, batch_size=5) as reader:
            batches = list(reader)

        assert batches == []

    def test_async_frame_reader_empty_video(self):
        """AsyncFrameReader should handle empty video gracefully."""
        mock_cap = MagicMock()
        mock_cap.read.return_value = (False, None)

        reader = AsyncFrameReader(mock_cap, batch_size=5)
        batches = list(reader)
        reader.close()

        assert batches == []

    def test_read_frames_async_yields_batches(self):
        """read_frames_async should yield frame batches."""
        mock_cap = MagicMock()
        frames = [np.zeros((50, 50, 3), dtype=np.uint8) for _ in range(5)]
        frame_index = [0]

        def mock_read():
            if frame_index[0] < len(frames):
                frame = frames[frame_index[0]]
                frame_index[0] += 1
                return True, frame
            return False, None

        mock_cap.read = mock_read

        batches = list(read_frames_async(mock_cap, batch_size=2))

        # Should have 3 batches: 2, 2, 1
        assert len(batches) == 3
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1

    def test_async_frame_reader_prefetches_next_batch(self):
        """AsyncFrameReader should prefetch the next batch while processing."""
        mock_cap = MagicMock()
        frames = [np.zeros((50, 50, 3), dtype=np.uint8) for _ in range(6)]
        frame_index = [0]

        def mock_read():
            if frame_index[0] < len(frames):
                frame = frames[frame_index[0]]
                frame_index[0] += 1
                return True, frame
            return False, None

        mock_cap.read = mock_read

        reader = AsyncFrameReader(mock_cap, batch_size=3)

        # Get first batch - next batch should already be prefetching
        batch1 = next(reader)
        assert len(batch1) == 3

        # Get second batch
        batch2 = next(reader)
        assert len(batch2) == 3

        reader.close()

    def test_async_frame_reader_handles_partial_last_batch(self):
        """AsyncFrameReader should handle partial last batch correctly."""
        mock_cap = MagicMock()
        frames = [np.zeros((50, 50, 3), dtype=np.uint8) for _ in range(7)]
        frame_index = [0]

        def mock_read():
            if frame_index[0] < len(frames):
                frame = frames[frame_index[0]]
                frame_index[0] += 1
                return True, frame
            return False, None

        mock_cap.read = mock_read

        batches = list(read_frames_async(mock_cap, batch_size=3))

        # Should have 3 batches: 3, 3, 1
        assert len(batches) == 3
        assert len(batches[-1]) == 1  # Last batch is partial
