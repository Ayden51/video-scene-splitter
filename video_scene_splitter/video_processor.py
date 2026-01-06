"""
Video processing functions for splitting videos using FFmpeg.

This module handles the actual video splitting operations, using FFmpeg
to create frame-accurate cuts at detected scene boundaries.
"""

import os
import subprocess
from pathlib import Path


def split_video_at_timestamps(video_path, scene_timestamps, output_dir):
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
    - Re-encoding with libx264 (H.264) for frame-accurate cuts
    - AAC audio encoding to preserve audio quality

    Output files are named: {original_name}_scene_{number}.{ext}
    Example: video_scene_001.mp4, video_scene_002.mp4, etc.

    Args:
        video_path (str): Path to the original video file.
        scene_timestamps (list): List of scene start times in seconds (float).
        output_dir (str): Directory where output files will be saved.

    Returns:
        int: Number of successfully created video files.

    Raises:
        None: Errors are printed but don't stop processing of remaining scenes.

    Side Effects:
        - Creates multiple video files in output_dir
        - Prints progress for each scene being processed
        - Prints summary when complete

    Note:
        Re-encoding is necessary for frame-accurate cuts. This means:
        - Processing takes longer than stream copying
        - Output files may be slightly larger or smaller than input
        - Quality is preserved with high-quality H.264 encoding
        - All scenes will have consistent encoding parameters

    Example:
        >>> timestamps = [0.0, 5.23, 12.45]
        >>> split_video_at_timestamps("video.mp4", timestamps, "output")
        Splitting video with frame-accurate precision...
        Scene 001: 0.00s → 5.23s
        Scene 002: 5.23s → 12.45s
        Scene 003: 12.45s → end
        ✓ Created 3 video files in 'output'
    """
    if len(scene_timestamps) < 1:
        print("No scenes detected.")
        return 0

    print("\n" + "=" * 70)
    print("Splitting video with frame-accurate precision...")
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
            "-c:v",
            "libx264",  # Re-encode for frame accuracy
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
    print("Note: Videos were re-encoded for frame-accurate cuts")

    return success_count
