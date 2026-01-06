"""
Unit tests for video_scene_splitter.video_processor module.

Tests the video splitting functionality with mocked FFmpeg calls.
"""

from unittest.mock import MagicMock, patch

from video_scene_splitter.video_processor import split_video_at_timestamps


class TestSplitVideoAtTimestamps:
    """Tests for split_video_at_timestamps function."""

    @patch("video_scene_splitter.video_processor.subprocess.run")
    def test_returns_zero_for_empty_timestamps(self, mock_run, temp_output_dir):
        """Should return 0 when no timestamps are provided."""
        result = split_video_at_timestamps("video.mp4", [], temp_output_dir)
        assert result == 0
        mock_run.assert_not_called()

    @patch("video_scene_splitter.video_processor.subprocess.run")
    def test_creates_correct_number_of_scenes(self, mock_run, temp_output_dir):
        """Should create one file per scene."""
        mock_run.return_value = MagicMock(returncode=0)
        timestamps = [0.0, 5.23, 12.45]

        result = split_video_at_timestamps("video.mp4", timestamps, temp_output_dir)

        assert result == 3
        assert mock_run.call_count == 3

    @patch("video_scene_splitter.video_processor.subprocess.run")
    def test_output_filenames_are_correct(self, mock_run, temp_output_dir):
        """Output files should have correct naming pattern."""
        mock_run.return_value = MagicMock(returncode=0)
        timestamps = [0.0, 5.23]

        split_video_at_timestamps("my_video.mp4", timestamps, temp_output_dir)

        calls = mock_run.call_args_list
        # Check that the output filename (last argument) contains the expected pattern
        assert "my_video_scene_001.mp4" in calls[0][0][0][-1]
        assert "my_video_scene_002.mp4" in calls[1][0][0][-1]

    @patch("video_scene_splitter.video_processor.subprocess.run")
    def test_preserves_video_extension(self, mock_run, temp_output_dir):
        """Should preserve the original video file extension."""
        mock_run.return_value = MagicMock(returncode=0)
        timestamps = [0.0, 5.0]

        split_video_at_timestamps("video.mkv", timestamps, temp_output_dir)

        calls = mock_run.call_args_list
        # Check that the output filename (last argument) contains the expected pattern
        assert "video_scene_001.mkv" in calls[0][0][0][-1]

    @patch("video_scene_splitter.video_processor.subprocess.run")
    def test_ffmpeg_command_structure(self, mock_run, temp_output_dir):
        """FFmpeg command should have correct structure."""
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
    def test_first_scene_start_time(self, mock_run, temp_output_dir):
        """First scene should start at the first timestamp."""
        mock_run.return_value = MagicMock(returncode=0)
        timestamps = [0.0, 5.23, 12.45]

        split_video_at_timestamps("video.mp4", timestamps, temp_output_dir)

        cmd = mock_run.call_args_list[0][0][0]
        ss_index = cmd.index("-ss")
        assert cmd[ss_index + 1] == "0.0"

    @patch("video_scene_splitter.video_processor.subprocess.run")
    def test_middle_scene_has_end_time(self, mock_run, temp_output_dir):
        """Middle scenes should have both start and end times."""
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
    def test_last_scene_no_end_time(self, mock_run, temp_output_dir):
        """Last scene should not have -to flag (goes to end of video)."""
        mock_run.return_value = MagicMock(returncode=0)
        timestamps = [0.0, 5.23, 12.45]

        split_video_at_timestamps("video.mp4", timestamps, temp_output_dir)

        # Check last scene (index 2)
        cmd = mock_run.call_args_list[2][0][0]
        assert "-to" not in cmd
        ss_index = cmd.index("-ss")
        assert cmd[ss_index + 1] == "12.45"

    @patch("video_scene_splitter.video_processor.subprocess.run")
    def test_handles_ffmpeg_failure(self, mock_run, temp_output_dir):
        """Should handle FFmpeg failures gracefully."""
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
    def test_single_scene(self, mock_run, temp_output_dir):
        """Should handle single scene correctly."""
        mock_run.return_value = MagicMock(returncode=0)
        timestamps = [0.0]

        result = split_video_at_timestamps("video.mp4", timestamps, temp_output_dir)

        assert result == 1
        assert mock_run.call_count == 1

    @patch("video_scene_splitter.video_processor.subprocess.run")
    def test_output_directory_in_path(self, mock_run, temp_output_dir):
        """Output files should be placed in the specified directory."""
        mock_run.return_value = MagicMock(returncode=0)
        timestamps = [0.0, 5.0]

        split_video_at_timestamps("video.mp4", timestamps, temp_output_dir)

        cmd = mock_run.call_args_list[0][0][0]
        output_file = cmd[-1]
        assert temp_output_dir in output_file

    @patch("video_scene_splitter.video_processor.subprocess.run")
    def test_overwrite_flag_present(self, mock_run, temp_output_dir):
        """FFmpeg command should include -y flag to overwrite files."""
        mock_run.return_value = MagicMock(returncode=0)
        timestamps = [0.0]

        split_video_at_timestamps("video.mp4", timestamps, temp_output_dir)

        cmd = mock_run.call_args_list[0][0][0]
        assert "-y" in cmd

    @patch("video_scene_splitter.video_processor.subprocess.run")
    def test_capture_output_enabled(self, mock_run, temp_output_dir):
        """subprocess.run should be called with capture_output=True."""
        mock_run.return_value = MagicMock(returncode=0)
        timestamps = [0.0]

        split_video_at_timestamps("video.mp4", timestamps, temp_output_dir)

        assert mock_run.call_args_list[0][1]["capture_output"] is True
