# Contributing to Video Scene Splitter

Thank you for your interest in contributing to Video Scene Splitter! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment Setup](#development-environment-setup)
- [Code Style and Formatting](#code-style-and-formatting)
- [Running Tests](#running-tests)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting Guidelines](#issue-reporting-guidelines)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors. Please be considerate and constructive in your communications.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/video-scene-splitter.git
   cd video-scene-splitter
   ```
3. **Add the upstream repository**:
   ```bash
   git remote add upstream https://github.com/Ayden51/video-scene-splitter.git
   ```

## Development Environment Setup

### Prerequisites

- Python 3.14 or higher
- FFmpeg installed and available in system PATH
- Git for version control

### Using uv (Recommended)

1. **Install uv**: Follow the [official uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for your platform, or use pip:
   ```bash
   pip install uv
   ```

2. **Create and activate virtual environment**:
   ```bash
   uv venv
   # Follow the activation command shown in the output
   ```

3. **Install all dependencies including development tools**:
   ```bash
   uv sync --all-extras
   ```

   This installs:
   - Core dependencies (NumPy, OpenCV)
   - Development tools (Ruff, pre-commit, Poe the Poet)

### Using pip (Alternative)

1. **Create virtual environment**:
   ```bash
   python -m venv venv
   ```

2. **Activate virtual environment**:
   - Windows (PowerShell): `.\venv\Scripts\Activate.ps1`
   - Windows (CMD): `.\venv\Scripts\activate.bat`
   - macOS/Linux: `source venv/bin/activate`

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Verify Installation

Test that everything is set up correctly:

```bash
# Using task runner scripts (recommended)
uv run poe --version

# Or run the main script directly
python main.py
```

If everything is set up correctly, the script should run without errors (though it may fail if no input video is provided).

## Code Style and Formatting

### Python Style Guidelines

- Follow [PEP 8](https://pep8.org/) style guide
- Use 4 spaces for indentation (no tabs)
- Maximum line length: 100 characters
- Use descriptive variable and function names
- Add docstrings to all functions, classes, and modules

### Docstring Format

Use Google-style docstrings:

```python
def detect_hard_cut(frame1, frame2, threshold):
    """Detect if there is a hard cut between two frames.
    
    Args:
        frame1: First frame as numpy array
        frame2: Second frame as numpy array
        threshold: Detection sensitivity threshold
        
    Returns:
        bool: True if hard cut detected, False otherwise
    """
    pass
```

### Code Organization

- Keep functions focused and single-purpose
- Prefer pure functions where possible
- Separate concerns (detection, processing, I/O)
- Use type hints for function parameters and return values

### Development Tools

This project uses **Ruff** for code formatting, import sorting, and linting. Ruff is an extremely fast all-in-one Python development tool written in Rust that replaces Black, isort, and Flake8 while being 10-100x faster.

We also use **Poe the Poet** as a task runner to provide npm-like scripts for common development tasks.

#### Installation

Install development dependencies using uv (recommended):

```bash
uv sync --all-extras
```

Or using pip:

```bash
pip install -e ".[dev]"
```

This will install:

- **ruff**: Fast linter and formatter
- **pre-commit**: Git hook framework for running checks before commits
- **poethepoet**: Task runner for development scripts

#### Using Task Runner Scripts (Recommended)

The easiest way to run development tasks is using the predefined scripts:

```bash
# Format code
uv run poe format

# Lint code
uv run poe lint

# Auto-fix linting issues
uv run poe lint-fix

# Run all checks (lint + format check)
uv run poe check

# Run the application
uv run poe start
```

To see all available scripts:

```bash
uv run poe
```

#### Running Ruff Directly

You can also run Ruff commands directly if preferred:

**Format code**:

```bash
ruff format .
```

**Lint code**:

```bash
ruff check .
```

**Lint with auto-fix**:

```bash
ruff check --fix .
```

**Check everything** (recommended before committing):

```bash
ruff check --fix . && ruff format .
```

#### Pre-commit Hooks (Recommended)

Set up pre-commit hooks to automatically run Ruff before each commit:

```bash
# Install pre-commit hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

Once installed, Ruff will automatically:

- Format your code
- Sort imports
- Check for common errors
- Fix issues automatically when possible

#### Editor Integration

**VS Code**: Install the [Ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)

Add to your `.vscode/settings.json`:

```json
{
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll": "explicit",
      "source.organizeImports": "explicit"
    }
  }
}
```

**PyCharm/IntelliJ**: Configure Ruff as an external tool or use the [Ruff plugin](https://plugins.jetbrains.com/plugin/20574-ruff)

**Other editors**: See [Ruff editor integrations](https://docs.astral.sh/ruff/integrations/)

#### Configuration

Ruff is configured in `pyproject.toml` with the following settings:

- Line length: 100 characters (matching PEP 8 guidelines)
- Target: Python 3.14+
- Enabled rules: pycodestyle, Pyflakes, isort, pep8-naming, pyupgrade, flake8-bugbear, and more
- Format style: Black-compatible

See `pyproject.toml` for the complete configuration.

## Running Tests

This project includes a comprehensive automated test suite with 83 tests achieving **99.42% code coverage**. All contributors must run tests before submitting pull requests.

### Prerequisites

Install development dependencies to run tests:

```bash
# Using uv (recommended)
uv sync --all-extras

# Or using pip
pip install -e ".[dev]"
```

This installs:

- **pytest**: Test framework
- **pytest-cov**: Coverage reporting plugin
- **pytest-xdist**: Parallel test execution
- **pytest-mock**: Mocking utilities for testing

### Running the Test Suite

#### Basic Test Execution

Run all tests:

```bash
# Using task runner (recommended)
uv run poe test

# Or using pytest directly
uv run pytest

# Or with pip installation
pytest
```

#### Verbose Output

For detailed test output showing each test name and result:

```bash
# Using task runner
uv run poe test-verbose

# Or using pytest directly
uv run pytest -vv
```

#### Coverage Reporting

**Always run tests with coverage before submitting a pull request:**

```bash
# Using task runner (recommended)
uv run poe test-coverage

# Or using pytest directly
uv run pytest --cov=video_scene_splitter --cov-report=term-missing --cov-report=html
```

This generates:

- **Terminal output**: Shows coverage percentages and lists missing lines
- **HTML report**: Detailed coverage report in `htmlcov/index.html` (open in browser)

#### Parallel Test Execution

Run tests faster using multiple CPU cores:

```bash
# Using task runner (recommended)
uv run poe test-parallel

# Or using pytest directly
uv run pytest -n auto
```

This automatically detects available CPU cores and distributes tests across them.

### Test Organization

Tests are organized by module and categorized with markers:

```
tests/
├── conftest.py              # Shared fixtures and pytest configuration
├── test_detection.py        # Scene detection algorithm tests (36 tests)
├── test_splitter.py         # VideoSceneSplitter integration tests (10 tests)
├── test_utils.py            # Utility function tests (24 tests)
└── test_video_processor.py  # Video processing tests (13 tests)
```

### Running Specific Tests

```bash
# Run tests from a specific file
pytest tests/test_detection.py

# Run a specific test class
pytest tests/test_detection.py::TestComputeHistogramDistance

# Run a specific test
pytest tests/test_detection.py::TestComputeHistogramDistance::test_identical_frames_return_zero_distance

# Run tests matching a pattern
pytest -k "histogram"

# Run tests by marker
pytest -m unit              # Only unit tests
pytest -m integration       # Only integration tests
pytest -m "not slow"        # Skip slow tests
```

### Test Markers

Tests are categorized using pytest markers:

| Marker | Description | Usage |
|--------|-------------|-------|
| `unit` | Unit tests for individual functions | `pytest -m unit` |
| `integration` | Integration tests for complete workflows | `pytest -m integration` |
| `slow` | Tests that take significant time to run | `pytest -m "not slow"` |

### Writing Tests

When contributing new features or fixing bugs, follow these testing guidelines:

#### Test Structure

Use the Arrange-Act-Assert pattern:

```python
def test_feature_name():
    """Clear description of what is being tested."""
    # Arrange: Set up test data and conditions
    input_data = create_test_data()

    # Act: Execute the function being tested
    result = function_under_test(input_data)

    # Assert: Verify the expected outcome
    assert result == expected_value
```

#### Test Naming

- Use descriptive names: `test_<function>_<scenario>_<expected_result>`
- Examples:
  - `test_compute_histogram_distance_identical_frames_return_zero`
  - `test_is_hard_cut_low_threshold_returns_false`
  - `test_split_video_empty_timestamps_returns_zero`

#### Test Coverage Guidelines

- **Minimum coverage**: 95% for new code
- **Current coverage**: 99.42% (maintain or improve)
- **Required tests**:
  - Happy path (normal operation)
  - Edge cases (boundary conditions)
  - Error cases (invalid inputs)
  - Integration scenarios (component interaction)

#### Using Fixtures

Leverage pytest fixtures for reusable test data:

```python
import pytest
import numpy as np

@pytest.fixture
def black_frame():
    """Create a black test frame."""
    return np.zeros((100, 100, 3), dtype=np.uint8)

def test_with_fixture(black_frame):
    """Test using the fixture."""
    assert black_frame.shape == (100, 100, 3)
```

See `tests/conftest.py` for available shared fixtures.

#### Mocking External Dependencies

Use `pytest-mock` for mocking external dependencies:

```python
from unittest.mock import Mock, patch

@patch("video_scene_splitter.splitter.cv2.VideoCapture")
def test_with_mock(mock_capture):
    """Test with mocked VideoCapture."""
    mock_cap = Mock()
    mock_cap.isOpened.return_value = True
    mock_capture.return_value = mock_cap

    # Test code here
```

### Before Submitting Pull Requests

**Required checks before submitting:**

1. **Run all tests**:
   ```bash
   uv run poe test
   ```

2. **Check coverage** (must maintain 95%+ coverage):
   ```bash
   uv run poe test-coverage
   ```

3. **Run code quality checks**:
   ```bash
   uv run poe check
   ```

4. **Run all checks together**:
   ```bash
   uv run poe check-all
   ```

All checks must pass before your pull request will be reviewed.

### Debugging Test Failures

If tests fail:

1. **Run with verbose output**:
   ```bash
   pytest -vv
   ```

2. **Run specific failing test**:
   ```bash
   pytest tests/test_file.py::TestClass::test_method -vv
   ```

3. **Use pytest debugging**:
   ```bash
   pytest --pdb  # Drop into debugger on failure
   ```

4. **Show local variables**:
   ```bash
   pytest --showlocals
   ```

5. **Disable output capture** to see print statements:
   ```bash
   pytest -s
   ```

### Manual Testing

In addition to automated tests, manually test your changes:

1. **Prepare test videos**: Place sample videos in the `input/` directory
2. **Run the application**: Execute `python main.py` or `uv run poe start`
3. **Verify output**: Check that scenes are detected and split correctly
4. **Test edge cases**: Try videos with various characteristics:
   - Different resolutions (480p, 720p, 1080p, 4K)
   - Different frame rates (24fps, 30fps, 60fps)
   - Fast cuts vs. slow transitions
   - Various video codecs (H.264, H.265, VP9)
   - Different containers (MP4, MKV, AVI, MOV)

### Continuous Integration

All tests run automatically on:

- Every pull request
- Every commit to the main branch
- Before releases

Pull requests must:

- Pass all tests
- Maintain or improve code coverage
- Pass all linting checks

## Making Changes

### Branch Naming

Create a descriptive branch name:

- Feature: `feature/add-new-detection-algorithm`
- Bug fix: `fix/incorrect-timestamp-calculation`
- Documentation: `docs/update-installation-guide`

### Commit Messages

Write clear, descriptive commit messages:

```
Add support for MKV video format

- Update video processor to handle MKV containers
- Add format detection in VideoSceneSplitter
- Update documentation with MKV examples
```

Format:

- First line: Brief summary (50 characters or less)
- Blank line
- Detailed description with bullet points if needed

### Keep Changes Focused

- One feature or fix per pull request
- Keep pull requests small and reviewable
- Update documentation alongside code changes

## Pull Request Process

### Before Submitting

1. **Update your fork**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run code quality checks**:
   ```bash
   # Using task runner scripts (recommended)
   uv run poe check

   # Or run commands directly
   ruff check --fix .
   ruff format .

   # Or use pre-commit to run all checks
   pre-commit run --all-files
   ```

3. **Test your changes thoroughly**:
   - Run the script with various test videos
   - Verify no regressions in existing functionality
   - Check that documentation is updated

4. **Review your changes**:
   ```bash
   git diff upstream/main
   ```

### Submitting the Pull Request

1. **Push to your fork**:
   ```bash
   git push origin your-branch-name
   ```

2. **Create pull request** on GitHub with:
   - Clear title describing the change
   - Detailed description of what changed and why
   - Reference any related issues (e.g., "Fixes #123")
   - Screenshots or examples if applicable

3. **Pull request template**:
   ```markdown
   ## Description
   Brief description of changes

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Performance improvement

   ## Testing
   Describe how you tested these changes

   ## Checklist
   - [ ] Code follows project style guidelines
   - [ ] Code formatted and linted (run `uv run poe check` or `ruff check --fix && ruff format`)
   - [ ] Pre-commit hooks pass (if installed)
   - [ ] Documentation updated
   - [ ] Changes tested with sample videos
   - [ ] No breaking changes (or documented if necessary)
   ```

### Review Process

- Maintainers will review your pull request
- Address any feedback or requested changes
- Once approved, your PR will be merged
- Your contribution will be acknowledged in the changelog

## Issue Reporting Guidelines

### Before Creating an Issue

1. **Search existing issues** to avoid duplicates
2. **Verify the issue** with the latest version
3. **Gather information**:
   - Python version
   - FFmpeg version
   - Operating system
   - Video file details (codec, resolution, frame rate)

### Creating a Bug Report

Use this template:

```markdown
## Bug Description
Clear description of the bug

## Steps to Reproduce
1. Step one
2. Step two
3. Step three

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: [e.g., Windows 11, macOS 14, Ubuntu 22.04]
- Python version: [e.g., 3.14.0]
- FFmpeg version: [e.g., 6.0]
- Package version: [e.g., 0.1.0]

## Video Details (if applicable)
- Format: [e.g., MP4, MKV]
- Codec: [e.g., H.264, H.265]
- Resolution: [e.g., 1920x1080]
- Frame rate: [e.g., 30fps]

## Additional Context
Any other relevant information, error messages, or screenshots
```

### Feature Requests

Use this template:

```markdown
## Feature Description
Clear description of the proposed feature

## Use Case
Why is this feature needed? What problem does it solve?

## Proposed Solution
How you envision this feature working

## Alternatives Considered
Other approaches you've thought about

## Additional Context
Any other relevant information or examples
```

### Questions and Discussions

For general questions:

- Check the README.md and documentation first
- Search existing issues and discussions
- Create a new issue with the "question" label

## Task Runner Scripts

This project uses **Poe the Poet** to provide npm-like scripts for common development tasks. This is a temporary convenience feature until a proper CLI interface is implemented.

### Available Scripts

| Script | Command | Description |
|--------|---------|-------------|
| `start` | `uv run poe start` | Run the main application |
| `lint` | `uv run poe lint` | Check code quality with Ruff |
| `format` | `uv run poe format` | Format code with Ruff |
| `lint-fix` | `uv run poe lint-fix` | Auto-fix linting issues |
| `check` | `uv run poe check` | Run all code quality checks (lint + format check) |

### Why Use Scripts?

- **Consistency**: Everyone on the team uses the same commands
- **Convenience**: Shorter commands that are easier to remember
- **Flexibility**: Easy to update commands without changing documentation
- **Familiarity**: Similar to npm scripts for developers from JavaScript backgrounds

### Future CLI Plans

These task runner scripts are a temporary solution. A proper CLI interface with argument parsing (using argparse, click, or similar) is planned for future releases. When implemented:

- The CLI will use `[project.scripts]` entry points in `pyproject.toml`
- Development scripts (`lint`, `format`, etc.) will remain in `[tool.poe.tasks]`
- The `start` script may be replaced by a proper CLI command like `video-scene-splitter`

## Development Tips

### Project Structure

```
video_scene_splitter/
├── video_scene_splitter/
│   ├── __init__.py          # Package initialization
│   ├── splitter.py          # Main VideoSceneSplitter class
│   ├── detection.py         # Scene detection algorithms
│   ├── utils.py             # File I/O and metrics utilities
│   └── video_processor.py   # FFmpeg operations
├── main.py                  # CLI interface
├── input/                   # Input videos
└── output/                  # Output scenes and debug files
```

### Adding New Detection Algorithms

1. Add the algorithm function to `detection.py`
2. Follow the existing function signatures
3. Update the detection logic in `splitter.py`
4. Document the algorithm in docstrings and README
5. Test with various video types

### Debugging Tips

- Use `debug=True` when calling `detect_scenes()`
- Check the generated CSV file for frame-by-frame metrics
- Examine before/after frames at cut points
- Use FFmpeg directly to verify video processing issues

## Getting Help

- **Documentation**: Check README.md and code docstrings
- **Issues**: Search or create GitHub issues
- **Discussions**: Use GitHub Discussions for general questions

## Recognition

Contributors will be acknowledged in:

- The project README
- Release notes and changelog
- Git commit history

Thank you for contributing to Video Scene Splitter! 🎬
