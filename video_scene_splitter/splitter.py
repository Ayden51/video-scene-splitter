"""
Main VideoSceneSplitter class that orchestrates scene detection and video splitting.

This module provides the high-level interface for the video scene splitter,
coordinating between detection algorithms, video processing, and utilities.
"""

from pathlib import Path

import cv2

from .detection import compute_histogram_distance, compute_pixel_difference, is_hard_cut
from .gpu_utils import (
    GPUInfo,
    ProcessorType,
    detect_cuda_gpu,
    print_gpu_status,
    select_processor,
)
from .utils import save_debug_frames, save_metrics_to_csv, save_timestamps_to_file
from .video_processor import split_video_at_timestamps


class VideoSceneSplitter:
    """
    Intelligent video scene splitter that detects and splits videos at hard cuts.

    This class analyzes video content frame-by-frame using multiple computer vision
    algorithms to identify hard cuts—moments where the entire visual content changes
    abruptly. It then splits the original video into separate files for each scene.

    Attributes:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where output files will be saved.
        threshold (float): Detection sensitivity threshold (0-100 scale).
        min_scene_duration (float): Minimum scene length in seconds.
        scene_timestamps (list): List of detected scene start times in seconds.

    Example:
        >>> splitter = VideoSceneSplitter(
        ...     video_path="input/video.mp4",
        ...     threshold=20.0,
        ...     min_scene_duration=0.5
        ... )
        >>> splitter.detect_scenes(debug=True)
        >>> splitter.split_video()
    """

    def __init__(
        self,
        video_path,
        output_dir="output",
        threshold=30.0,
        min_scene_duration=1.5,
        processor="auto",
    ):
        """
        Initialize the video scene splitter for hard cut detection.

        Args:
            video_path (str): Path to the input video file (relative or absolute).
                Supported formats: MP4, AVI, MOV, MKV, and other OpenCV-compatible formats.
            output_dir (str, optional): Directory to save split videos and debug files.
                Defaults to "output". Created automatically if it doesn't exist.
            threshold (float, optional): Hard cut detection sensitivity threshold.
                Defaults to 30.0. Typical range: 15-40.
                - Lower values (15-20): More sensitive, may detect subtle changes
                - Medium values (20-30): Balanced, good for most videos
                - Higher values (30-40): Conservative, only obvious hard cuts
                - Values above 40: Very conservative, may miss some cuts
            min_scene_duration (float, optional): Minimum scene duration in seconds.
                Defaults to 1.5. Prevents detection of very short scenes.
                Useful for filtering out brief flashes or transitions.
            processor (str, optional): Processing mode for scene detection.
                Defaults to "auto". Options:
                - "auto": Automatically detect and use GPU if available, fallback to CPU
                - "cpu": Force CPU-only processing (useful for debugging or compatibility)
                - "gpu": Force GPU processing (raises error if GPU unavailable)

        Raises:
            ValueError: If video_path doesn't exist or threshold is negative.
            RuntimeError: If processor="gpu" but no GPU is available.

        Note:
            The threshold operates on a 0-100 scale where higher values indicate
            greater difference between frames. Run with debug=True to see statistics
            and get a suggested threshold for your specific video.
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.threshold = threshold
        self.min_scene_duration = min_scene_duration
        self.scene_timestamps = []

        # GPU detection and processor selection
        self._gpu_info: GPUInfo = detect_cuda_gpu()
        self._processor_request = ProcessorType.from_string(processor)
        self._active_processor: ProcessorType = select_processor(
            self._processor_request, self._gpu_info
        )

    def detect_scenes(self, debug=False):
        """
        Detect hard cuts (scene changes) in the video using multi-metric analysis.

        This method analyzes the video frame-by-frame, comparing consecutive frames
        using histogram distance, pixel difference, and changed pixel ratio. A hard
        cut is detected when all three metrics exceed their thresholds simultaneously,
        indicating an abrupt change in visual content.

        The detection process:
        1. Reads video frames sequentially
        2. Computes three metrics for each frame pair:
           - HSV histogram distance (color content change)
           - Mean pixel difference (brightness/intensity change)
           - Changed pixel ratio (spatial change coverage)
        3. Identifies hard cuts when all conditions are met:
           - Histogram distance > threshold
           - Pixel difference > threshold
           - Changed pixel ratio > 20%
           - Time since last cut >= min_scene_duration
        4. Records timestamps of detected scene boundaries

        Args:
            debug (bool, optional): Enable debug mode for detailed analysis.
                Defaults to False. When True:
                - Displays detailed GPU hardware information and acceleration status
                - Saves before/after frames at each detected cut as JPEG images
                - Generates CSV file with frame-by-frame metrics
                - Prints statistical analysis (min, max, mean, std dev)
                - Suggests optimal threshold based on video statistics

        Returns:
            list: List of scene start timestamps in seconds (float).
                Always includes 0.0 as the first scene start.
                Example: [0.0, 5.23, 12.45, 18.67]

        Raises:
            ValueError: If video file cannot be opened or first frame cannot be read.

        Example:
            >>> splitter = VideoSceneSplitter("video.mp4", threshold=25.0)
            >>> timestamps = splitter.detect_scenes(debug=True)
            Analyzing video: video.mp4
            Video: 30.00 FPS, 900 frames, 30.00s
            ✓ Scene 2 at 5.23s (frame 157)
            ✓ Scene 3 at 12.45s (frame 374)
            >>> print(timestamps)
            [0.0, 5.23, 12.45]

        Note:
            Progress is displayed every 100 frames. For long videos, this provides
            feedback that processing is ongoing. The final output includes scene
            durations and statistics when debug mode is enabled.
        """
        print(f"Analyzing video: {self.video_path}")
        Path(self.output_dir).mkdir(exist_ok=True)

        # Print GPU/processor status (detailed in debug mode, brief otherwise)
        print_gpu_status(self._gpu_info, self._active_processor, debug=debug)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        min_frame_gap = int(self.min_scene_duration * fps)

        print(f"Video: {fps:.2f} FPS, {total_frames} frames, {duration:.2f}s")
        print(f"Threshold: {self.threshold}")
        print(f"Min scene duration: {self.min_scene_duration}s ({min_frame_gap} frames)")

        # Read first frame
        ret, prev_frame = cap.read()
        if not ret:
            raise ValueError("Cannot read first frame")

        frame_num = 1
        last_scene_frame = 0
        self.scene_timestamps = [0.0]

        all_metrics = []

        print("\n" + "=" * 70)
        print("Detecting hard cuts...")
        print("=" * 70)

        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break

            # Compute multiple metrics using detection algorithms
            hist_dist = compute_histogram_distance(prev_frame, curr_frame)
            pixel_diff, changed_ratio = compute_pixel_difference(prev_frame, curr_frame)

            metrics = {
                "frame": frame_num,
                "timestamp": frame_num / fps,
                "hist_dist": hist_dist,
                "pixel_diff": pixel_diff,
                "changed_ratio": changed_ratio * 100,  # Convert to percentage
            }
            all_metrics.append(metrics)

            # Check for hard cut using detection algorithm
            frames_since_last = frame_num - last_scene_frame

            if is_hard_cut(
                hist_dist,
                pixel_diff,
                changed_ratio,
                self.threshold,
                frames_since_last,
                min_frame_gap,
            ):
                timestamp = frame_num / fps
                self.scene_timestamps.append(timestamp)

                print(
                    f"✓ Scene {len(self.scene_timestamps)} at {timestamp:.2f}s (frame {frame_num})"
                )
                print(
                    f"  Hist: {hist_dist:.1f}, Pixel: {pixel_diff:.1f}, "
                    f"Changed: {changed_ratio * 100:.1f}%"
                )

                last_scene_frame = frame_num

                # Save debug frames if requested
                if debug:
                    save_debug_frames(
                        self.output_dir,
                        len(self.scene_timestamps),
                        frame_num - 1,
                        frame_num,
                        prev_frame,
                        curr_frame,
                    )

            prev_frame = curr_frame.copy()
            frame_num += 1

            # Progress indicator
            if frame_num % 100 == 0:
                progress = (frame_num / total_frames) * 100
                print(f"Progress: {progress:.1f}%", end="\r")

        cap.release()

        print("\n" + "=" * 70)
        print(f"✓ Detected {len(self.scene_timestamps)} scenes")

        # Print scene durations
        if len(self.scene_timestamps) > 1:
            print("\nScene durations:")
            for i in range(len(self.scene_timestamps) - 1):
                dur = self.scene_timestamps[i + 1] - self.scene_timestamps[i]
                print(f"  Scene {i + 1}: {dur:.2f}s")

            last_dur = duration - self.scene_timestamps[-1]
            print(f"  Scene {len(self.scene_timestamps)}: {last_dur:.2f}s")

        # Save analysis data if debug mode is enabled
        if debug:
            save_metrics_to_csv(all_metrics, self.output_dir)

        return self.scene_timestamps

    def save_timestamps(self, filename="timestamps.txt"):
        """
        Save detected scene timestamps to a text file.

        Creates a human-readable text file containing all detected scene start
        times along with metadata about the video and detection parameters.
        Useful for reference, manual review, or importing into other tools.

        Args:
            filename (str, optional): Name of the output file.
                Defaults to "timestamps.txt". The file is saved in output_dir.

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
            >>> splitter.save_timestamps("my_scenes.txt")
            ✓ Timestamps saved to output/my_scenes.txt
        """
        save_timestamps_to_file(
            self.video_path, self.scene_timestamps, self.threshold, self.output_dir, filename
        )

    def split_video(self):
        """
        Split video into separate files at detected scene boundaries.

        This method uses FFmpeg to split the original video into multiple files,
        one for each detected scene. The splitting is frame-accurate, achieved by
        re-encoding the video with H.264 codec. Each output file contains one
        complete scene from start to end (or to the next scene boundary).

        Returns:
            int: Number of successfully created video files.

        Example:
            >>> splitter.detect_scenes()
            >>> count = splitter.split_video()
            Splitting video with frame-accurate precision...
            Scene 001: 0.00s → 5.23s
            Scene 002: 5.23s → 12.45s
            ✓ Created 2 video files in 'output'
        """
        return split_video_at_timestamps(self.video_path, self.scene_timestamps, self.output_dir)
