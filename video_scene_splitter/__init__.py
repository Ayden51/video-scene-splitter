"""
Video Scene Splitter - Automatic hard cut detection and video segmentation.

This package provides intelligent video scene detection using computer vision techniques
including HSV histogram analysis, pixel difference comparison, and changed pixel ratio
analysis to identify hard cuts (abrupt scene changes) in video content.

GPU acceleration is supported via NVIDIA CUDA when CuPy is installed.
"""

from .gpu_utils import GPUConfig, GPUInfo, ProcessorType, detect_cuda_gpu
from .splitter import VideoSceneSplitter

__version__ = "0.2.0-dev"
__all__ = [
    "GPUConfig",
    "GPUInfo",
    "ProcessorType",
    "VideoSceneSplitter",
    "detect_cuda_gpu",
]
