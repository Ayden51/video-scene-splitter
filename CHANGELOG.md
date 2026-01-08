# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Hybrid CPU/GPU processing (Phase 2B) - AUTO mode**
  - `AutoModeConfig` dataclass for configuring hybrid processing behavior
    - `min_resolution_for_gpu`: Minimum resolution to use GPU (default: 720p)
    - `use_gpu_for_pixel_diff`: GPU for pixel diff on HD+ (default: True)
    - `use_gpu_for_histogram`: CPU for histogram (default: False - 1.35x faster)
    - Resolution-based batch size caps (SD=60, HD=30, 4K=15 frames)
  - `select_operation_processor()`: Per-operation processor selection function
    - Returns CPU for histogram computation (always - benchmarked 1.35x faster)
    - Returns GPU for pixel diff on HD+ content (≥720p - benchmarked 1.29x faster)
    - Returns CPU for SD content (transfer overhead too high)
  - `get_resolution_batch_size_cap()`: Resolution-aware batch size selection
  - `_detect_scenes_hybrid()`: New hybrid detection pipeline in VideoSceneSplitter
    - Automatically selects optimal processor per operation based on resolution
    - GPU pixel diff + CPU histogram for HD+ content
    - Full CPU processing for SD content (optimal for small frames)
    - GPU OOM error handling with automatic CPU fallback
  - `compute_pixel_difference_batch_cpu()`: CPU batch pixel difference for hybrid mode
  - `compute_histogram_distance_batch_cpu()`: CPU batch histogram for hybrid mode
  - Updated routing logic in `detect_scenes()` to dispatch to hybrid mode for AUTO

- **GPU-accelerated scene detection algorithms (Phase 2A)**
  - `compute_pixel_difference_gpu()`: Single frame pair pixel difference on GPU
  - `compute_histogram_distance_gpu()`: Single frame pair histogram distance on GPU
  - `compute_pixel_difference_batch_gpu()`: Batch processing for pixel difference (5-120 frames)
  - `compute_histogram_distance_batch_gpu()`: Batch processing for histogram distance
  - `histogram_correlation_batch_gpu()`: Vectorized batch correlation without Python loops
  - `bgr_to_gray_gpu()` and `bgr_to_hsv_gpu()`: Color space conversion helpers
  - `free_gpu_memory()`: Explicit GPU memory cleanup function

- **Enhanced debug mode with detailed GPU hardware information**
  - Debug mode (`debug=True`) now displays comprehensive GPU hardware details including:
    - GPU name, memory (total and free), CUDA version, driver version
    - Compute capability and backend information
    - Processor mode and acceleration status
  - Normal mode (`debug=False`) shows brief, user-friendly status messages
  - Helps users verify GPU detection and troubleshoot hardware acceleration issues
  - Updated `print_gpu_status()` function to accept `debug` parameter for conditional output

- **GPU benchmarking infrastructure**
  - `benchmarks/gpu_benchmark.py`: Comprehensive GPU vs CPU performance testing
  - Support for SD, HD, and 4K video benchmarking
  - Batch size variation testing (5, 15, 30, 60 frames)
  - Results documentation in `benchmarks/results/`

- **45 new GPU detection tests** in `tests/test_detection_gpu.py`
  - Single frame pair GPU tests for pixel difference and histogram distance
  - Batch processing tests with various sizes (2, 5, 10, 15, 30, 61 frames)
  - GPU vs CPU result consistency tests (tolerance: pixel 1e-5, histogram rel=0.3)
  - GPU memory management tests
  - Color space conversion tests (BGR to Gray, BGR to HSV)

### Changed

- **Improved GPU histogram performance** from 0.32x to 0.77x of CPU speed
  - Vectorized batch histogram correlation (removed per-pair Python loops)
  - Added explicit memory cleanup with `del` statements
  - GPU histogram now competitive at 0.7-0.8x of CPU (was 0.3x before)

### Performance

- **GPU Pixel Difference**: 1.19-1.43x speedup for HD video (1080p)
- **GPU Histogram Distance**: 0.69-0.77x of CPU (improved from 0.32x)
- **Overall**: GPU provides meaningful speedup for HD pixel difference; histogram is acceptable

### Technical Notes

- **Hybrid processing strategy** (AUTO mode):
  - Histogram: Always CPU (1.35x faster than GPU due to transfer overhead)
  - Pixel diff: GPU for HD+ (≥720p), CPU for SD (transfer overhead dominates)
  - This strategy provides optimal performance across all video resolutions
- GPU detection algorithms require CuPy with CUDA 13.0+
- Histogram counting still requires loop (scatter operation limitation)
- Full histogram parallelization requires custom CUDA kernels (planned for v0.3.0+)

## [0.1.1] - 2026-01-06

This patch release improves documentation, fixes configuration defaults, and enhances the user experience for new users.

### Changed

#### Documentation Improvements

- **Restructured README.md Usage section** for better clarity and user-friendliness
  - Added **Basic Setup** subsection with step-by-step instructions to create `input/` folder and place video files
  - Added **Configuration** subsection explaining how to edit `main.py` to specify video path
  - Split **Execution** section into two clear subsections:
    - **With Scripts**: Using task runner (`uv run poe start`)
    - **Without Scripts**: Direct Python execution (`python main.py`)
  - Added **Output** subsection clearly stating that results are saved in `output/` folder
  - Updated **Programmatic Usage** examples to use correct default values (threshold=30.0, min_scene_duration=1.5)
  - Kept **Advanced Usage** examples for fine-tuned detection scenarios

- **Reorganized README.md section order** for improved information flow
  - New order: Features → Requirements → Installation → Usage → Output Files → Configuration Options → Troubleshooting → How It Works → License → Contributing → Support
  - Moved **Output Files** section before **Configuration Options** for better logical flow
  - Removed **Testing** section from README.md (comprehensive testing documentation remains in CONTRIBUTING.md)

#### Configuration Fixes

- **Fixed main.py default configuration** to match VideoSceneSplitter class defaults
  - Changed `video_path` from specific file `"input/06-nhung-em-be-bot.mp4"` to generic `"input/sample_video.mp4"`
  - Reset `threshold` from `20.0` to default value `30.0`
  - Reset `min_scene_duration` from `0.5` to default value `1.5` seconds
  - Updated docstring comments to reflect default values

### Added

- **Added `.augment/` to .gitignore** to exclude Augment-specific files from version control

### Notes

- This release focuses on improving the user experience for new users by providing clearer documentation and correct default configurations
- All changes are non-breaking and backward compatible
- Testing documentation has been consolidated in CONTRIBUTING.md for better organization
- Default values in `main.py` now correctly match the VideoSceneSplitter class constructor defaults

## [0.1.0] - 2026-01-06

Initial release of Video Scene Splitter - an intelligent video processing tool that automatically detects and splits videos at hard cuts using computer vision techniques.

### Added

#### Core Video Processing Features

- **Automatic hard cut detection** using multiple computer vision algorithms
  - HSV histogram analysis for color content comparison between frames
  - Pixel-level difference detection for measuring direct frame changes
  - Changed pixel ratio analysis to determine percentage of significantly altered pixels
- **Frame-accurate video splitting** with FFmpeg integration
  - Precise cuts at exact frame boundaries
  - H.264 video encoding with AAC audio for optimal compatibility
  - Support for multiple video formats (MP4, AVI, MOV, MKV)
- **Configurable sensitivity controls**
  - Adjustable threshold parameter (15-40 typical range) for detection sensitivity
  - Minimum scene duration setting to filter out very short scenes
  - Fine-tuning options for different video types and content
- **Debug mode** with comprehensive analysis tools
  - Automatic saving of before/after frames at each detected cut point
  - Detailed metrics CSV export with frame-by-frame analysis data
  - Statistical analysis with threshold recommendations
  - Visual verification capabilities for detection accuracy
- **Progress tracking** with real-time updates during video analysis
- **Timestamp export** functionality
  - Save detected scene timestamps to text files
  - Include metadata about detection parameters and video information
  - Support for further processing and integration with other tools

#### Package Architecture

- **Modular package structure** for maintainability and reusability
  - `video_scene_splitter.splitter`: Main VideoSceneSplitter class with high-level API
  - `video_scene_splitter.detection`: Scene detection algorithms and metrics calculation
  - `video_scene_splitter.utils`: File I/O operations and metrics utilities
  - `video_scene_splitter.video_processor`: FFmpeg operations and video manipulation
- **Clean separation** between library code and CLI interface
  - Usable as both a Python library and command-line tool
  - Importable modules for integration into other projects
  - Standalone CLI script for direct usage
- **Comprehensive documentation** with usage examples
  - Detailed README with installation instructions
  - Code examples for basic and advanced usage
  - Troubleshooting guide for common issues
  - API documentation in docstrings

#### Technical Highlights

- **Pure functions** for easy testing and composition
  - Functional programming approach where applicable
  - Predictable behavior with no side effects
  - Easy to unit test and verify
- **Separation of concerns** across modules
  - Detection logic isolated from video processing
  - I/O operations separated from core algorithms
  - Clear interfaces between components
- **Dual-mode operation**
  - Library mode: Import and use in Python scripts
  - CLI mode: Run directly from command line
- **Extensible design** for adding new detection methods
  - Plugin-friendly architecture
  - Easy to add new algorithms without modifying existing code
  - Clear patterns for extending functionality

#### Testing Infrastructure

- **Comprehensive test suite** with 83 automated tests
  - **99.42% code coverage** across all modules
  - Unit tests for individual functions and algorithms
  - Integration tests for complete workflows
  - Edge case validation and error handling tests
  - Test fixtures for reusable test data
- **Testing task runner scripts** using Poe the Poet
  - `uv run poe test`: Run the full test suite
  - `uv run poe test-verbose`: Run tests with detailed output
  - `uv run poe test-coverage`: Run tests with coverage reporting
  - `uv run poe test-parallel`: Run tests in parallel for faster execution
  - `uv run poe check-all`: Run all checks including tests
- **Testing dependencies** in dev dependencies
  - pytest >= 8.3.0: Test framework
  - pytest-cov >= 6.0.0: Coverage reporting
  - pytest-xdist >= 3.6.0: Parallel test execution
  - pytest-mock >= 3.14.0: Mocking utilities

#### Development Tools

- **Task runner scripts** using Poe the Poet for npm-like development workflow
  - `uv run poe start`: Run the main application
  - `uv run poe lint`: Check code quality with Ruff
  - `uv run poe format`: Format code with Ruff
  - `uv run poe lint-fix`: Auto-fix linting issues
  - `uv run poe check`: Run all code quality checks
- **Development dependency**: poethepoet >= 0.31.1 for task running
- **Code quality tools**:
  - Ruff >= 0.8.0: Fast Python linter and formatter
  - pre-commit >= 4.0.0: Git hook framework

### Project Configuration

- **Python 3.14+** requirement for modern language features
- **Core dependencies**
  - OpenCV (opencv-python) >= 4.11.0.86 for computer vision operations
  - NumPy >= 2.4.0 for numerical computing and array operations
  - FFmpeg (system dependency) for video processing
- **Package management**
  - Support for both pip and uv package managers
  - uv recommended for faster and more reliable dependency resolution
  - Standard pyproject.toml configuration
  - Requirements.txt for traditional pip installations
- **Comprehensive README** with detailed documentation
  - Installation instructions for multiple platforms (Windows, macOS, Linux)
  - Usage examples from basic to advanced scenarios
  - Configuration options and parameter guidelines
  - Troubleshooting section for common issues
  - Technical details about detection algorithms
- **Standard Python package structure**
  - pyproject.toml for modern Python packaging
  - Proper package initialization and imports
  - Clean directory structure
  - Version management

### Documentation

- **README.md** with comprehensive documentation
  - Installation guide with platform-specific instructions (Windows, macOS, Linux)
  - Usage examples for common scenarios (basic and advanced)
  - Testing section with coverage reporting and test execution
  - Available scripts documentation
  - API reference with parameter descriptions
  - Configuration options and threshold guidelines
  - Troubleshooting guide for common issues
  - Algorithm explanation and technical details
- **CONTRIBUTING.md** with detailed contributor guidelines
  - Development environment setup instructions
  - Code style and formatting requirements (Ruff configuration)
  - Comprehensive testing guidelines
    - Prerequisites and setup
    - Test execution examples
    - Writing new tests
    - Coverage expectations (95%+ for new code)
    - Debugging test failures
  - Task runner scripts documentation
  - Pull request process
  - Issue reporting templates
- **MIT License** with proper copyright notice

### Notes

- This is the initial release of Video Scene Splitter
- Task runner scripts are a temporary convenience feature until a proper CLI interface is implemented
- When CLI is implemented, it will use `[project.scripts]` entry points in pyproject.toml
- Development scripts will remain in `[tool.poe.tasks]` for consistency
- All pull requests must pass the full test suite and maintain 95%+ code coverage

[0.1.1]: https://github.com/Ayden51/video-scene-splitter/releases/tag/v0.1.1
[0.1.0]: https://github.com/Ayden51/video-scene-splitter/releases/tag/v0.1.0
