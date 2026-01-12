"""
Video processing functions for splitting videos using FFmpeg.

This module handles the actual video splitting operations, using FFmpeg
to create frame-accurate cuts at detected scene boundaries. Supports both
CPU-based encoding (libx264) and GPU-accelerated NVENC encoding when
available on NVIDIA hardware.

Includes multi-threaded frame reading for improved performance during
GPU-accelerated scene detection.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from .gpu_utils import ProcessorType


# =============================================================================
# NVENC Hardware Encoding Support
# =============================================================================

_nvenc_available: bool | None = None  # Cached detection result


def detect_nvenc_support() -> bool:
    """
    Detect if NVENC hardware encoding is available.

    Checks if FFmpeg has h264_nvenc encoder support by querying FFmpeg's
    encoder list. The result is cached for subsequent calls.

    Returns:
        bool: True if NVENC is available, False otherwise.

    Note:
        NVENC requires:
        - NVIDIA GPU with NVENC support (most GPUs since GTX 600 series)
        - Appropriate NVIDIA drivers installed
        - FFmpeg compiled with NVENC support
    """
    global _nvenc_available

    if _nvenc_available is not None:
        return _nvenc_available

    try:
        # Query FFmpeg for available encoders
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Check if h264_nvenc is in the encoder list
        _nvenc_available = "h264_nvenc" in result.stdout
        return _nvenc_available

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        _nvenc_available = False
        return False


def get_encoder_options(
    processor: ProcessorType | str | None = None,
    quality_preset: str = "p4",
) -> list[str]:
    """
    Get FFmpeg encoder options based on processor mode and availability.

    The encoder selection follows these rules:
    - "cpu": Always use libx264 (software encoding)
    - "gpu": Prefer NVENC, raise error if unavailable
    - "auto": Use NVENC if available, fallback to libx264
    - None: Same as "auto"

    NVENC provides 3-8x faster encoding compared to libx264 with similar quality.

    Args:
        processor: Processing mode ("cpu", "gpu", "auto", or None for auto).
            Can also be a ProcessorType enum value.
        quality_preset: NVENC quality preset (p1-p7, where p1 is fastest
            and p7 is highest quality). Default "p4" provides balanced
            speed/quality. Only used when NVENC is selected.

    Returns:
        list[str]: FFmpeg encoder arguments (e.g., ["-c:v", "h264_nvenc", ...])

    Raises:
        RuntimeError: If processor="gpu" and NVENC is not available.

    Example:
        >>> get_encoder_options("auto")
        ['-c:v', 'h264_nvenc', '-preset', 'p4', '-rc', 'vbr', '-cq', '23', '-b:v', '0']
        >>> get_encoder_options("cpu")
        ['-c:v', 'libx264']
    """
    # Normalize processor type to string
    if processor is not None and hasattr(processor, "value"):
        processor = processor.value  # Convert ProcessorType enum to string

    processor_str = (processor or "auto").lower()

    # CPU mode: always use software encoding
    if processor_str == "cpu":
        return _get_libx264_options()

    # Check NVENC availability for GPU or AUTO modes
    nvenc_available = detect_nvenc_support()

    if processor_str == "gpu":
        # GPU mode: require NVENC
        if not nvenc_available:
            raise RuntimeError(
                "GPU encoding requested but NVENC is not available.\n"
                "Possible causes:\n"
                "  - No NVIDIA GPU with NVENC support\n"
                "  - NVIDIA drivers not installed\n"
                "  - FFmpeg not compiled with NVENC support\n\n"
                "To use CPU encoding instead, set processor='cpu' or processor='auto'"
            )
        return _get_nvenc_options(quality_preset)

    # AUTO mode: use NVENC if available, otherwise libx264
    if nvenc_available:
        return _get_nvenc_options(quality_preset)
    return _get_libx264_options()


def _get_nvenc_options(preset: str = "p4") -> list[str]:
    """
    Get NVENC encoder options for high-quality hardware encoding.

    Args:
        preset: Quality preset (p1-p7). Default "p4" is balanced.

    Returns:
        list[str]: FFmpeg NVENC encoder arguments.
    """
    return [
        "-c:v",
        "h264_nvenc",
        "-preset",
        preset,  # p1=fastest, p4=balanced, p7=quality
        "-rc",
        "vbr",  # Variable bitrate mode
        "-cq",
        "23",  # Quality level (similar to CRF, lower=better)
        "-b:v",
        "0",  # Let encoder determine bitrate based on quality
    ]


def _get_libx264_options() -> list[str]:
    """
    Get libx264 encoder options for software encoding.

    Returns:
        list[str]: FFmpeg libx264 encoder arguments.
    """
    return ["-c:v", "libx264"]


def get_encoder_info(processor: ProcessorType | str | None = None) -> dict[str, str | bool]:
    """
    Get information about the encoder that would be used.

    Useful for displaying encoder status to users and for debugging.

    Args:
        processor: Processing mode ("cpu", "gpu", "auto", or None).

    Returns:
        dict: Encoder information with keys:
            - "name": Encoder name (e.g., "h264_nvenc" or "libx264")
            - "type": "hardware" or "software"
            - "nvenc_available": Whether NVENC is available
            - "selected_processor": The effective processor mode
    """
    if processor is not None and hasattr(processor, "value"):
        processor = processor.value

    processor_str = (processor or "auto").lower()
    nvenc_available = detect_nvenc_support()

    if processor_str == "cpu":
        encoder_name = "libx264"
        encoder_type = "software"
    elif processor_str == "gpu":
        encoder_name = "h264_nvenc" if nvenc_available else "unavailable"
        encoder_type = "hardware" if nvenc_available else "unavailable"
    else:  # auto
        if nvenc_available:
            encoder_name = "h264_nvenc"
            encoder_type = "hardware"
        else:
            encoder_name = "libx264"
            encoder_type = "software"

    return {
        "name": encoder_name,
        "type": encoder_type,
        "nvenc_available": nvenc_available,
        "selected_processor": processor_str,
    }


def split_video_at_timestamps(
    video_path: str,
    scene_timestamps: list[float],
    output_dir: str,
    processor: ProcessorType | str | None = None,
) -> int:
    """
    Split video into separate files at detected scene boundaries.

    This function uses FFmpeg to split the original video into multiple files,
    one for each detected scene. The splitting is frame-accurate, achieved by
    re-encoding the video with H.264 codec. Each output file contains one
    complete scene from start to end (or to the next scene boundary).

    The function uses FFmpeg with optimized parameters:
    - Input file specified first for proper stream handling
    - Start time (-ss) placed after input for frame accuracy
    - End time (-to) for precise endpoint (more accurate than -t duration)
    - Re-encoding with H.264 for frame-accurate cuts (NVENC or libx264)
    - AAC audio encoding to preserve audio quality

    Encoder selection based on processor mode:
    - "cpu": Always uses libx264 (software encoding)
    - "gpu": Requires NVENC hardware encoding (raises error if unavailable)
    - "auto" or None: Uses NVENC if available, otherwise falls back to libx264

    Output files are named: {original_name}_scene_{number}.{ext}
    Example: video_scene_001.mp4, video_scene_002.mp4, etc.

    Args:
        video_path: Path to the original video file.
        scene_timestamps: List of scene start times in seconds (float).
        output_dir: Directory where output files will be saved.
        processor: Processing mode for encoder selection.
            - "cpu": Force libx264 software encoding
            - "gpu": Force NVENC hardware encoding (error if unavailable)
            - "auto" or None: Use NVENC if available, else libx264

    Returns:
        int: Number of successfully created video files.

    Raises:
        RuntimeError: If processor="gpu" and NVENC is not available.

    Side Effects:
        - Creates multiple video files in output_dir
        - Prints progress for each scene being processed
        - Prints encoder information when starting
        - Prints summary when complete

    Note:
        Re-encoding is necessary for frame-accurate cuts. This means:
        - Processing takes longer than stream copying
        - Output files may be slightly larger or smaller than input
        - Quality is preserved with high-quality H.264 encoding
        - All scenes will have consistent encoding parameters
        - NVENC provides 3-8x faster encoding than libx264

    Example:
        >>> timestamps = [0.0, 5.23, 12.45]
        >>> split_video_at_timestamps("video.mp4", timestamps, "output")
        Splitting video with frame-accurate precision...
        Encoder: h264_nvenc (hardware)
        Scene 001: 0.00s → 5.23s
        Scene 002: 5.23s → 12.45s
        Scene 003: 12.45s → end
        ✓ Created 3 video files in 'output'
    """
    if len(scene_timestamps) < 1:
        print("No scenes detected.")
        return 0

    # Get encoder options based on processor mode
    encoder_options = get_encoder_options(processor)
    encoder_info = get_encoder_info(processor)

    print("\n" + "=" * 70)
    print("Splitting video with frame-accurate precision...")
    print(f"Encoder: {encoder_info['name']} ({encoder_info['type']})")
    print("=" * 70)

    base_name = Path(video_path).stem
    video_ext = Path(video_path).suffix
    success_count = 0

    for i in range(len(scene_timestamps)):
        start_time = scene_timestamps[i]

        if i < len(scene_timestamps) - 1:
            # Not the last scene - set end time to next scene's start
            duration_flag = ["-to", str(scene_timestamps[i + 1])]
            end_info = f"{scene_timestamps[i + 1]:.2f}s"
        else:
            # Last scene - process until end of video
            duration_flag = []
            end_info = "end"

        output_file = os.path.join(output_dir, f"{base_name}_scene_{i + 1:03d}{video_ext}")

        # Frame-accurate cutting: -i first, then -ss for accurate seeking
        # Using -to instead of -t for more precise end point
        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-ss",
            str(start_time),
            *duration_flag,
            *encoder_options,  # Dynamic encoder selection (NVENC or libx264)
            "-c:a",
            "aac",  # Re-encode audio
            "-strict",
            "experimental",
            "-avoid_negative_ts",
            "1",
            "-y",  # Overwrite output files
            output_file,
        ]

        print(f"Scene {i + 1:03d}: {start_time:.2f}s → {end_info}")
        result = subprocess.run(cmd, capture_output=True)

        if result.returncode == 0:
            success_count += 1
        else:
            print(f"  ⚠ Error encoding scene {i + 1}")

    print("=" * 70)
    print(f"✓ Created {success_count} video files in '{output_dir}'")
    if encoder_info["type"] == "hardware":
        print("Note: Videos were re-encoded with NVENC hardware acceleration")
    else:
        print("Note: Videos were re-encoded for frame-accurate cuts")

    return success_count


# =============================================================================
# Multi-threaded Frame Reading
# =============================================================================


def _read_batch(cap: cv2.VideoCapture, batch_size: int) -> list[np.ndarray]:
    """
    Read a batch of frames from video capture.

    Internal helper function used by read_frames_async to read frames
    in a separate thread.

    Args:
        cap: OpenCV VideoCapture object.
        batch_size: Maximum number of frames to read.

    Returns:
        list: List of frames (numpy arrays). May be shorter than batch_size
            if the end of video is reached, or empty if no frames remain.
    """
    frames = []
    for _ in range(batch_size):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames


class AsyncFrameReader:
    """
    Asynchronous frame reader for overlapping I/O with GPU computation.

    Uses a ThreadPoolExecutor with a single worker to read frames in the
    background while the main thread processes the previous batch on GPU.
    This implements a producer-consumer pattern with double buffering.

    The reader prefetches the next batch while the current batch is being
    processed, hiding I/O latency and improving overall throughput.

    Attributes:
        cap: OpenCV VideoCapture object.
        batch_size: Number of frames per batch.
        executor: ThreadPoolExecutor for async operations.
        future: Future object for the pending read operation.
        video_ended: Flag indicating if the video has ended.

    Example:
        >>> reader = AsyncFrameReader(cap, batch_size=30)
        >>> for batch in reader:
        ...     # Process batch on GPU while next batch is being read
        ...     results = process_on_gpu(batch)
        >>> reader.close()
    """

    def __init__(self, cap: cv2.VideoCapture, batch_size: int = 30):
        """
        Initialize the async frame reader.

        Args:
            cap: OpenCV VideoCapture object (must already be opened).
            batch_size: Number of frames to read per batch. Default is 30.
        """
        self.cap = cap
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._future: Future[list[np.ndarray]] | None = None
        self._video_ended = False
        # Start prefetching the first batch immediately
        self._start_prefetch()

    def _start_prefetch(self) -> None:
        """Start reading the next batch in the background."""
        if not self._video_ended:
            self._future = self.executor.submit(_read_batch, self.cap, self.batch_size)

    def __iter__(self) -> Iterator[list[np.ndarray]]:
        """Iterate over frame batches."""
        return self

    def __next__(self) -> list[np.ndarray]:
        """
        Get the next batch of frames.

        Waits for the prefetched batch and starts prefetching the next one.

        Returns:
            list: Batch of frames.

        Raises:
            StopIteration: When no more frames are available.
        """
        if self._video_ended and self._future is None:
            raise StopIteration

        # Get the prefetched batch
        batch = self._future.result() if self._future is not None else []

        # Check if video has ended
        if len(batch) < self.batch_size:
            self._video_ended = True
            self._future = None
        else:
            # Start prefetching next batch
            self._start_prefetch()

        if not batch:
            raise StopIteration

        return batch

    def close(self) -> None:
        """
        Shutdown the executor and release resources.

        Should be called when done reading frames to clean up threads.
        """
        self.executor.shutdown(wait=False)

    def __enter__(self) -> AsyncFrameReader:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensures executor is shutdown."""
        self.close()


def read_frames_async(
    cap: cv2.VideoCapture,
    batch_size: int = 30,
) -> Iterator[list[np.ndarray]]:
    """
    Read frames using a dedicated thread for I/O.

    This function overlaps CPU I/O with GPU computation, hiding latency.
    It implements a producer-consumer pattern with double buffering.

    The reader prefetches the next batch while the current batch is being
    processed, providing 10-20% overall speedup when combined with GPU
    processing.

    Args:
        cap: OpenCV VideoCapture object (must already be opened).
        batch_size: Number of frames per batch. Default is 30.
            Should match the GPU batch size for optimal performance.

    Yields:
        list: Batch of frames (numpy arrays). The last batch may be
            shorter than batch_size if the video ends mid-batch.

    Example:
        >>> cap = cv2.VideoCapture("video.mp4")
        >>> for batch in read_frames_async(cap, batch_size=30):
        ...     # Process batch on GPU
        ...     gpu_results = process_on_gpu(batch)
        >>> cap.release()

    Note:
        For best performance, the batch_size should match the GPU batch
        size used for processing. This ensures the GPU is kept busy while
        the next batch of frames is being read.

        The AsyncFrameReader class provides additional control (close method,
        context manager support) if needed.
    """
    reader = AsyncFrameReader(cap, batch_size)
    try:
        yield from reader
    finally:
        reader.close()
