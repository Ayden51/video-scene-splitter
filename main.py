"""
Video Scene Splitter - Main entry point and CLI interface.

This module provides the command-line interface for the Video Scene Splitter.
It demonstrates basic usage of the VideoSceneSplitter class and can be
customized for different video processing needs.
"""

from video_scene_splitter import VideoSceneSplitter


def main():
    """
    Main entry point for the Video Scene Splitter application.

    This function demonstrates basic usage of the VideoSceneSplitter class.
    It processes a video file with predefined parameters, detects scenes,
    saves timestamps, and splits the video into separate files.

    Configuration:
        - video_path: Path to input video (modify this for your video)
        - output_dir: Directory for output files
        - threshold: Detection sensitivity (30.0 = default)
        - min_scene_duration: Minimum scene length (1.5s = default)
        - debug: Enabled for detailed analysis and statistics

    Workflow:
        1. Initialize VideoSceneSplitter with parameters
        2. Detect scenes with debug mode enabled
        3. Save timestamps to text file
        4. Split video into separate scene files

    Note:
        Modify the configuration variables to process your own videos.
        Run with debug=True first to see statistics and determine
        the optimal threshold for your specific video content.
    """
    # Configuration
    video_path = "input/sample_video.mp4"  # Change this to your video
    output_dir = "output"
    threshold = 30.0  # Default: 30.0 (typical range: 15-40)
    min_scene_duration = 1.5  # Default: 1.5 seconds
    debug = False  # Set to True for detailed analysis and debug images

    # Initialize the video scene splitter
    splitter = VideoSceneSplitter(
        video_path=video_path,
        output_dir=output_dir,
        threshold=threshold,
        min_scene_duration=min_scene_duration,
    )

    # Run detection
    splitter.detect_scenes(debug=debug)

    # Save timestamps to a text file
    splitter.save_timestamps()

    # Split the video into separate scene files
    splitter.split_video()

    # Print completion message
    if debug:
        print("\n✓ All done! Check the debug images to verify cut detection.")
    else:
        print("\n✓ All done! Video has been split into scenes.")


if __name__ == "__main__":
    main()
