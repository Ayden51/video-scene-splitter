# Agent Guidelines for Video Scene Splitter

This document provides essential information for AI coding agents working on the Video Scene Splitter project.

## Project Overview

A Python 3.14+ video processing tool that detects hard cuts in videos using computer vision (OpenCV, NumPy) and splits them into scenes using FFmpeg. The project achieves 99.42% test coverage with 83 tests.

## Build, Lint, and Test Commands

### Setup

```bash
# Using uv (recommended)
uv sync --all-extras

# Or using pip
pip install -e ".[dev]"
```

### Linting and Formatting

```bash
# Format code (run before committing)
uv run poe format
# Or: ruff format .

# Check linting issues
uv run poe lint
# Or: ruff check .

# Auto-fix linting issues
uv run poe lint-fix
# Or: ruff check --fix .

# Run all code quality checks
uv run poe check
# Or: ruff check --fix . && ruff format .
```

### Testing

```bash
# Run all tests
uv run poe test
# Or: pytest

# Run tests with verbose output
uv run poe test-verbose
# Or: pytest -vv

# Run tests with coverage (REQUIRED before PRs)
uv run poe test-coverage
# Or: pytest --cov=video_scene_splitter --cov-report=term-missing --cov-report=html

# Run tests in parallel (faster)
uv run poe test-parallel
# Or: pytest -n auto

# Run a single test file
pytest tests/test_detection.py

# Run a single test class
pytest tests/test_detection.py::TestComputeHistogramDistance

# Run a single test function
pytest tests/test_detection.py::TestComputeHistogramDistance::test_identical_frames_return_zero_distance

# Run tests matching a pattern
pytest -k "histogram"

# Run tests by marker
pytest -m unit              # Only unit tests
pytest -m integration       # Only integration tests
pytest -m "not slow"        # Skip slow tests
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks (recommended)
pre-commit install

# Run all pre-commit checks manually
pre-commit run --all-files
```

### Running the Application

```bash
uv run poe start
# Or: python main.py
```

## Project Structure

```
video_scene_splitter/
├── video_scene_splitter/    # Main package
│   ├── __init__.py         # Package initialization
│   ├── splitter.py         # VideoSceneSplitter class (main interface)
│   ├── detection.py        # Scene detection algorithms
│   ├── utils.py            # File I/O and metrics utilities
│   └── video_processor.py  # FFmpeg video operations
├── tests/                  # Test suite (83 tests, 99.42% coverage)
│   ├── conftest.py         # Shared pytest fixtures
│   ├── test_detection.py   # Detection algorithm tests
│   ├── test_splitter.py    # VideoSceneSplitter tests
│   ├── test_utils.py       # Utility function tests
│   └── test_video_processor.py  # Video processing tests
├── main.py                 # CLI entry point
├── pyproject.toml          # Project configuration
└── README.md               # User documentation
```

## Code Style Guidelines

### General Python Style

- **Style Guide**: Follow PEP 8
- **Line Length**: Maximum 100 characters
- **Indentation**: 4 spaces (no tabs)
- **Target Version**: Python 3.14+
- **Tool**: Ruff (replaces Black, isort, Flake8)

### Import Organization

Imports are automatically sorted by Ruff with the following order:

1. Standard library imports
2. Third-party imports (numpy, cv2)
3. First-party imports (video_scene_splitter modules)

Example:
```python
from pathlib import Path

import cv2
import numpy as np

from .detection import compute_histogram_distance
from .utils import save_debug_frames
```

### Type Hints

- Use type hints for function parameters and return values
- Prefer built-in types over typing module when possible (Python 3.14+)

Example:
```python
def compute_histogram_distance(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Compute distance between frames."""
    pass
```

### Docstrings

Use **Google-style docstrings** for all public functions, classes, and modules:

```python
def detect_hard_cut(frame1, frame2, threshold):
    """Detect if there is a hard cut between two frames.
    
    Args:
        frame1 (numpy.ndarray): First frame in BGR format.
        frame2 (numpy.ndarray): Second frame in BGR format.
        threshold (float): Detection sensitivity threshold.
        
    Returns:
        bool: True if hard cut detected, False otherwise.
        
    Raises:
        ValueError: If frames have different dimensions.
    """
    pass
```

### Naming Conventions

- **Functions/Variables**: `snake_case` (e.g., `compute_histogram_distance`, `frame_num`)
- **Classes**: `PascalCase` (e.g., `VideoSceneSplitter`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_THRESHOLD`)
- **Private**: Leading underscore (e.g., `_internal_helper`)

### Error Handling

- Raise `ValueError` for invalid inputs
- Provide descriptive error messages
- Document exceptions in docstrings

Example:
```python
if not cap.isOpened():
    raise ValueError(f"Cannot open video: {self.video_path}")
```

### Code Organization

- **Keep functions focused**: Single responsibility principle
- **Prefer pure functions**: Avoid side effects where possible
- **Separate concerns**: Detection logic separate from I/O operations
- **Use meaningful names**: Avoid abbreviations unless universally understood

## Testing Guidelines

### Coverage Requirements

- **Minimum coverage**: 95% for new code
- **Current coverage**: 99.42% (maintain or improve)
- **Always run coverage before submitting PRs**

### Test Structure

Use the **Arrange-Act-Assert** pattern:

```python
def test_compute_histogram_distance_identical_frames():
    """Test that identical frames return zero distance."""
    # Arrange: Set up test data
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Act: Execute function
    distance = compute_histogram_distance(frame, frame)
    
    # Assert: Verify result
    assert distance == 0.0
```

### Test Naming

Format: `test_<function>_<scenario>_<expected_result>`

Examples:
- `test_compute_histogram_distance_identical_frames_return_zero`
- `test_is_hard_cut_below_threshold_returns_false`
- `test_split_video_empty_timestamps_returns_zero`

### Using Fixtures

Available shared fixtures in `tests/conftest.py`:
- `black_frame`: 100x100 black frame
- `white_frame`: 100x100 white frame
- `red_frame`, `blue_frame`, `green_frame`: Colored frames
- `gradient_frame`: Frame with horizontal gradient
- `noisy_black_frame`: Black frame with random noise
- `similar_frames`: Two nearly identical frames
- `temp_output_dir`: Temporary directory for test outputs

### Test Markers

Categorize tests using pytest markers:

```python
@pytest.mark.unit
def test_histogram_calculation():
    """Unit test for histogram calculation."""
    pass

@pytest.mark.integration
def test_full_video_processing():
    """Integration test for complete workflow."""
    pass

@pytest.mark.slow
def test_large_video_processing():
    """Slow test that processes large files."""
    pass
```

## Pre-commit Configuration

The project uses pre-commit hooks to ensure code quality:

1. **Ruff linting** with auto-fix
2. **Ruff formatting**
3. **File checks** (large files, merge conflicts, YAML/TOML validation)
4. **Trailing whitespace removal**
5. **End-of-file fixer**
6. **No direct commits to main branch**

Install with: `pre-commit install`

## Before Submitting Pull Requests

**Required checklist:**

1. Run all tests: `uv run poe test`
2. Check coverage: `uv run poe test-coverage` (must be ≥95%)
3. Run code quality checks: `uv run poe check`
4. Or run everything: `uv run poe check-all`
5. Ensure pre-commit hooks pass: `pre-commit run --all-files`

## Common Patterns

### Adding New Detection Algorithms

1. Add function to `video_scene_splitter/detection.py`
2. Follow existing function signatures (accept frames, return metrics)
3. Add comprehensive docstring with examples
4. Write unit tests in `tests/test_detection.py`
5. Update integration logic in `splitter.py` if needed

### Working with OpenCV

- Frames are in **BGR format** (not RGB)
- Use `cv2.cvtColor()` for color space conversions
- Frame shape is `(height, width, channels)`
- Always check return values from `cap.read()`

Example:
```python
ret, frame = cap.read()
if not ret:
    raise ValueError("Cannot read frame")
```

### Video Processing with FFmpeg

- Use `video_processor.py` for all FFmpeg operations
- The project uses frame-accurate splitting with re-encoding
- Timestamps are in seconds (float)
