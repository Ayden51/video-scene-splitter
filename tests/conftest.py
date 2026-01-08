"""
Pytest configuration and shared fixtures for Video Scene Splitter tests.

This module provides common fixtures and test utilities used across
all test modules.
"""

import numpy as np
import pytest

# =============================================================================
# GPU Test Support
# =============================================================================


def _gpu_available() -> bool:
    """Check if CUDA GPU is available for testing."""
    try:
        import cupy as cp

        _ = cp.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def gpu_available():
    """Session-scoped fixture indicating GPU availability."""
    return _gpu_available()


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu: Tests requiring CUDA GPU")


def pytest_collection_modifyitems(config, items):
    """Auto-skip GPU tests when GPU is not available."""
    del config  # Unused but required by pytest hook signature
    if _gpu_available():
        return  # GPU available, don't skip anything

    skip_gpu = pytest.mark.skip(reason="CUDA GPU not available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture
def black_frame():
    """Create a 100x100 black frame (all zeros)."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def white_frame():
    """Create a 100x100 white frame (all 255s)."""
    return np.full((100, 100, 3), 255, dtype=np.uint8)


@pytest.fixture
def red_frame():
    """Create a 100x100 red frame (BGR: 0, 0, 255)."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :, 2] = 255  # Red channel in BGR
    return frame


@pytest.fixture
def blue_frame():
    """Create a 100x100 blue frame (BGR: 255, 0, 0)."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :, 0] = 255  # Blue channel in BGR
    return frame


@pytest.fixture
def green_frame():
    """Create a 100x100 green frame (BGR: 0, 255, 0)."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :, 1] = 255  # Green channel in BGR
    return frame


@pytest.fixture
def gradient_frame():
    """Create a 100x100 frame with a horizontal gradient."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        frame[:, i, :] = int(i * 2.55)  # 0 to 255 gradient
    return frame


@pytest.fixture
def noisy_black_frame():
    """Create a 100x100 mostly black frame with small random noise."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # Add small random noise (0-10 range)
    noise = np.random.randint(0, 10, (100, 100, 3), dtype=np.uint8)
    return np.clip(frame + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def similar_frames():
    """Create two very similar frames (slight brightness difference)."""
    frame1 = np.full((100, 100, 3), 100, dtype=np.uint8)
    frame2 = np.full((100, 100, 3), 105, dtype=np.uint8)
    return frame1, frame2


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory for tests."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return str(output_dir)


# =============================================================================
# GPU Detection Test Fixtures
# =============================================================================


@pytest.fixture
def frame_batch_small():
    """Create a small batch of 5 frames for GPU batch processing tests.

    Returns list of 5 frames with varying content to test batch detection:
    - Frame 0: Black
    - Frame 1: Dark gray (similar to frame 0)
    - Frame 2: White (different from frames 0-1)
    - Frame 3: Gray (different from frame 2)
    - Frame 4: Black (different from frame 3)
    """
    frames = [
        np.zeros((100, 100, 3), dtype=np.uint8),  # Black
        np.full((100, 100, 3), 20, dtype=np.uint8),  # Dark gray
        np.full((100, 100, 3), 255, dtype=np.uint8),  # White
        np.full((100, 100, 3), 128, dtype=np.uint8),  # Gray
        np.zeros((100, 100, 3), dtype=np.uint8),  # Black
    ]
    return frames


@pytest.fixture
def frame_batch_medium():
    """Create a medium batch of 15 frames for GPU batch processing tests.

    Returns list of 15 frames alternating between similar and different content.
    """
    frames = []
    for i in range(15):
        # Alternate between black, gray, and white
        brightness = (i % 3) * 127  # 0, 127, 254
        frames.append(np.full((100, 100, 3), brightness, dtype=np.uint8))
    return frames


@pytest.fixture
def frame_batch_large():
    """Create a large batch of 31 frames for GPU batch processing tests.

    Returns list of 31 frames (30 pairs for batch processing).
    Simulates a typical batch_size=30 scenario.
    """
    frames = []
    np.random.seed(42)  # Reproducible random frames
    for i in range(31):
        # Create frames with varying brightness and some random variation
        base_brightness = (i * 8) % 256
        frame = np.full((100, 100, 3), base_brightness, dtype=np.uint8)
        # Add some random noise to make it more realistic
        noise = np.random.randint(-10, 10, (100, 100, 3))
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frames.append(frame)
    return frames


@pytest.fixture
def scene_change_frames():
    """Create frames simulating a scene change for GPU detection tests.

    Returns list of 10 frames:
    - Frames 0-4: Similar (same scene)
    - Frame 5: Dramatic change (scene cut)
    - Frames 5-9: Similar (new scene)
    """
    frames = []
    # First scene: blue-ish frames
    for i in range(5):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 0] = 200 + i  # Blue channel
        frame[:, :, 1] = 50 + i  # Green channel
        frame[:, :, 2] = 50 + i  # Red channel
        frames.append(frame)

    # Second scene: red-ish frames (dramatic change)
    for i in range(5):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 0] = 50 + i  # Blue channel
        frame[:, :, 1] = 50 + i  # Green channel
        frame[:, :, 2] = 200 + i  # Red channel
        frames.append(frame)

    return frames


@pytest.fixture
def identical_frame_batch():
    """Create a batch of identical frames for testing zero-difference case."""
    frame = np.full((100, 100, 3), 100, dtype=np.uint8)
    return [frame.copy() for _ in range(10)]
