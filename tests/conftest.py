"""
Pytest configuration and shared fixtures for Video Scene Splitter tests.

This module provides common fixtures and test utilities used across
all test modules.
"""

import numpy as np
import pytest


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
