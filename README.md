# Video Scene Splitter

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![Package Version](https://img.shields.io/badge/version-0.1.1-green.svg)](https://github.com/Ayden51/video-scene-splitter)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://docs.astral.sh/uv/)

A video processing tool that automatically detects and splits videos at hard cuts using computer vision techniques. It analyzes video content frame-by-frame to identify hard cuts—moments where the visual content changes abruptly, such as transitions between different camera angles, locations, or subjects—and splits the original video into separate files for each detected scene.

## ✨ Features

- **Intelligent Hard Cut Detection**: Uses multiple computer vision algorithms to accurately identify scene changes
  - HSV histogram analysis for color content comparison
  - Pixel-level difference detection
  - Changed pixel ratio analysis
- **Frame-Accurate Video Splitting**: Precise cuts at exact frame boundaries using FFmpeg
- **Configurable Sensitivity**: Adjustable threshold and minimum scene duration parameters
- **Debug Mode**:
  - Saves before/after frames at each detected cut
  - Generates detailed metrics CSV with frame-by-frame analysis
  - Provides statistical analysis and threshold recommendations
- **Progress Tracking**: Real-time progress updates during video analysis
- **Timestamp Export**: Saves scene timestamps for reference and further processing

## 📋 Requirements

- **Python**: 3.14 or higher
- **FFmpeg**: Must be installed and available in system PATH
- **Dependencies**:
  - OpenCV (opencv-python) >= 4.11.0.86
  - NumPy >= 2.4.0

## 🚀 Installation

### Prerequisites

Before installing the project dependencies, ensure you have the following installed:

1. **Python 3.14+**: Download from [python.org](https://www.python.org/downloads/) or use your system's package manager
2. **FFmpeg**: Follow the [official FFmpeg installation guide](https://ffmpeg.org/download.html) for your platform

Verify installations:

```bash
python --version  # Should show 3.14 or higher
ffmpeg -version   # Should display FFmpeg version info
```

### Setting Up the Project

#### Using uv (Recommended)

[uv](https://docs.astral.sh/uv/) is a fast, reliable Python package manager that provides better dependency resolution and faster installations.

1. **Install uv**: Follow the [official uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for your platform, or use pip:
   ```bash
   pip install uv
   ```

2. **Create a virtual environment**:
   ```bash
   uv venv
   ```

   **Note**: Follow the activation command shown in the output from `uv venv`, as it may differ based on your shell (bash, zsh, PowerShell, etc.).

3. **Install dependencies**:
   ```bash
   uv sync
   ```

4. **Install development tools** (optional, for using task runner scripts):
   ```bash
   uv sync --all-extras
   ```

   This installs additional development tools including:
   - **Poe the Poet**: Task runner for npm-like scripts
   - **Ruff**: Fast Python linter and formatter
   - **pre-commit**: Git hook framework

#### Using pip (Alternative)

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

2. **Activate the virtual environment**:

   - **Windows (PowerShell)**:
     ```bash
     .\venv\Scripts\Activate.ps1
     ```

   - **Windows (Command Prompt)**:
     ```bash
     .\venv\Scripts\activate.bat
     ```

   - **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Or install packages directly:
   ```bash
   pip install opencv-python>=4.11.0.86 numpy>=2.4.0
   ```

## 📖 Usage

### Basic Setup

1. **Create the input directory** (if it doesn't exist):
   ```bash
   mkdir input
   ```

2. **Place your video file** in the `input/` directory:
   ```bash
   # Example: Copy your video to the input folder
   cp /path/to/your/video.mp4 input/
   ```

### Configuration

Edit `main.py` to specify your video file path:

```python
# Configuration
video_path = "input/your_video.mp4"  # Change this to your video filename
output_dir = "output"
threshold = 30.0  # Default: 30.0 (typical range: 15-40)
min_scene_duration = 1.5  # Default: 1.5 seconds
debug = False  # Set to True for detailed analysis and debug images
```

### Execution

#### With Scripts

If you installed the development tools (`uv sync --all-extras`), you can use convenient task runner scripts:

```bash
# Run the application
uv run poe start
```

**Available Scripts:**

| Script | Command | Description |
|--------|---------|-------------|
| `start` | `uv run poe start` | Run the main application (equivalent to `python main.py`) |
| `test` | `uv run poe test` | Run the test suite |
| `test-coverage` | `uv run poe test-coverage` | Run tests with coverage reporting |
| `test-parallel` | `uv run poe test-parallel` | Run tests in parallel for faster execution |
| `lint` | `uv run poe lint` | Check code quality with Ruff |
| `format` | `uv run poe format` | Format code with Ruff |
| `lint-fix` | `uv run poe lint-fix` | Auto-fix linting issues |
| `check` | `uv run poe check` | Run all code quality checks |
| `check-all` | `uv run poe check-all` | Run all checks including tests |

To see all available scripts:

```bash
uv run poe
```

**Note**: These scripts are a temporary convenience feature. A proper CLI interface with argument parsing is planned for future releases.

#### Without Scripts

Run the script directly using Python:

```bash
python main.py
```

Or with uv:

```bash
uv run main.py
```

### Output

After running the script, all results will be saved in the `output/` folder:

- **Video files**: `{original_name}_scene_001.mp4`, `{original_name}_scene_002.mp4`, etc.
- **Timestamp file**: `timestamps.txt` with scene start times
- **Debug files** (if debug mode is enabled): Before/after frames and metrics CSV

### Programmatic Usage

```python
from video_scene_splitter import VideoSceneSplitter

# Create splitter instance
splitter = VideoSceneSplitter(
    video_path="input/my_video.mp4",
    output_dir="output",
    threshold=30.0,
    min_scene_duration=1.5
)

# Detect scenes (with debug mode enabled)
splitter.detect_scenes(debug=True)

# Save timestamps to file
splitter.save_timestamps()

# Split video into separate files
splitter.split_video()
```

### Advanced Usage

```python
# Fine-tuned for sensitive detection
splitter = VideoSceneSplitter(
    video_path="input/fast_cuts.mp4",
    output_dir="output_fast",
    threshold=15.0,        # Lower = more sensitive
    min_scene_duration=0.3  # Shorter minimum scene length
)

# Detect without debug output
timestamps = splitter.detect_scenes(debug=False)

# Only save timestamps (no video splitting)
splitter.save_timestamps("my_timestamps.txt")
```

## 📁 Output Files

After running the script, you'll find the following in your output directory:

### Video Files

- `{original_name}_scene_001.mp4` - First scene
- `{original_name}_scene_002.mp4` - Second scene
- `{original_name}_scene_NNN.mp4` - Nth scene

### Timestamp File

- `timestamps.txt` - List of all scene start times with metadata

### Debug Files (when debug=True)

- `cut_001_before_fXXXX.jpg` - Frame before first cut
- `cut_001_after_fXXXX.jpg` - Frame after first cut
- `cut_detection_metrics.csv` - Detailed frame-by-frame analysis data

### Metrics CSV Format

```csv
Frame,Timestamp,Hist_Distance,Pixel_Diff,Changed_Ratio%
1,0.03,5.23,12.45,3.21
2,0.07,6.12,13.89,3.45
...
```

## ⚙️ Configuration Options

### VideoSceneSplitter Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video_path` | str | Required | Path to the input video file (relative or absolute) |
| `output_dir` | str | `"output"` | Directory where output files will be saved |
| `threshold` | float | `30.0` | Hard cut detection sensitivity (15-40 typical range) |
| `min_scene_duration` | float | `1.5` | Minimum scene duration in seconds |

### Threshold Guidelines

- **15-20**: Very sensitive - detects subtle scene changes, may produce false positives
- **20-30**: Balanced - good for most videos with clear scene transitions
- **30-40**: Conservative - only detects very obvious hard cuts
- **40+**: Very conservative - may miss some legitimate scene changes

**Tip**: Run with `debug=True` first to see the statistics and suggested threshold for your specific video.

### Method Parameters

#### `detect_scenes(debug=False)`

Analyzes the video and detects scene changes.

- **debug** (bool): When `True`, saves debug frames and detailed metrics
  - Before/after frames at each cut point
  - CSV file with frame-by-frame metrics
  - Statistical analysis and threshold recommendations

**Returns**: List of timestamps (in seconds) where scenes begin

#### `save_timestamps(filename="timestamps.txt")`

Saves detected scene timestamps to a text file.

- **filename** (str): Name of the output file (saved in `output_dir`)

#### `split_video()`

Splits the video into separate files at detected scene boundaries.

- Uses FFmpeg for frame-accurate cutting
- Re-encodes video with H.264 codec for precision
- Preserves audio with AAC encoding

## 🔧 Troubleshooting

### "Cannot open video" Error

**Problem**: Video file cannot be read by OpenCV

**Solutions**:

- Verify the file path is correct
- Ensure the video format is supported (MP4, AVI, MOV, MKV)
- Try re-encoding the video with FFmpeg:
  ```bash
  ffmpeg -i input.mp4 -c:v libx264 -c:a aac output.mp4
  ```

### FFmpeg Not Found

**Problem**: `ffmpeg` command not available

**Solutions**:

- Install FFmpeg (see Installation section)
- Verify installation: `ffmpeg -version`
- Add FFmpeg to your system PATH

### Scripts Not Available

**Problem**: `uv run poe` commands not working

**Solutions**:

- Install development dependencies: `uv sync --all-extras`
- Verify Poe the Poet is installed: `uv run poe --version`
- Alternatively, run commands directly: `python main.py` instead of `uv run poe start`

### Too Many/Too Few Scenes Detected

**Problem**: Detection is too sensitive or not sensitive enough

**Solutions**:

- **Too many scenes**: Increase `threshold` (try 30-40)
- **Too few scenes**: Decrease `threshold` (try 15-20)
- Run with `debug=True` to see suggested threshold
- Adjust `min_scene_duration` to filter out very short scenes

### Memory Issues with Large Videos

**Problem**: Script crashes or runs out of memory

**Solutions**:

- Process video in smaller chunks
- Reduce video resolution before processing
- Close other applications to free up RAM

### Inaccurate Cut Points

**Problem**: Scenes are cut at wrong timestamps

**Solutions**:

- The script uses frame-accurate cutting with re-encoding
- Verify FFmpeg is working correctly
- Check that the original video doesn't have variable frame rate (VFR)
- Convert to constant frame rate (CFR) if needed:
  ```bash
  ffmpeg -i input.mp4 -vsync cfr -c:v libx264 -c:a aac output.mp4
  ```

## 🛠️ How It Works

### Detection Algorithm

The scene detection uses a multi-metric approach:

1. **HSV Histogram Analysis**: Compares color distribution between consecutive frames
   - Converts frames to HSV color space (better for content comparison)
   - Calculates histograms for Hue, Saturation, and Value channels
   - Uses correlation comparison (weighted: H=50%, S=30%, V=20%)

2. **Pixel Difference**: Measures direct pixel-level changes
   - Converts frames to grayscale
   - Computes absolute difference between frames
   - Calculates mean difference value

3. **Changed Pixel Ratio**: Determines percentage of significantly changed pixels
   - Counts pixels with difference > 30 (on 0-255 scale)
   - Calculates ratio of changed pixels to total pixels

### Hard Cut Criteria

A hard cut is detected when ALL conditions are met:

- Histogram distance > threshold
- Pixel difference > threshold
- Changed pixel ratio > 20%
- Time since last cut >= minimum scene duration

### Video Splitting

Uses FFmpeg with frame-accurate parameters:

- Input file specified with `-i`
- Start time with `-ss` (after input for accuracy)
- End time with `-to` (more precise than `-t`)
- Re-encodes with H.264/AAC for frame accuracy

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Đặng Công Huy

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on:

- Setting up your development environment
- Code style and formatting requirements
- Running tests
- Submitting pull requests
- Reporting issues

Feel free to submit issues or pull requests to help improve this project!

## 📧 Support

For questions or issues, please open an issue on the project repository.
