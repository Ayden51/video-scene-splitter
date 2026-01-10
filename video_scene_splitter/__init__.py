"""
Video Scene Splitter - Automatic hard cut detection and video segmentation.

This package provides intelligent video scene detection using computer vision techniques
including HSV histogram analysis, pixel difference comparison, and changed pixel ratio
analysis to identify hard cuts (abrupt scene changes) in video content.

GPU acceleration is supported via NVIDIA CUDA when CuPy is installed.
"""

from .gpu_utils import (
    AutoModeConfig,
    GPUConfig,
    GPUInfo,
    ProcessorType,
    detect_cuda_gpu,
    estimate_optimal_batch_size,
    get_resolution_batch_size_cap,
    select_operation_processor,
    should_use_async_io,
)
from .splitter import VideoSceneSplitter
from .video_processor import (
    HardwareDecodeInfo,
    HardwareVideoReader,
    detect_nvdec_support,
    get_decode_info,
    is_codec_nvdec_compatible,
)

__version__ = "0.2.0-dev"
__all__ = [
    "AutoModeConfig",
    "GPUConfig",
    "GPUInfo",
    "HardwareDecodeInfo",
    "HardwareVideoReader",
    "ProcessorType",
    "VideoSceneSplitter",
    "detect_cuda_gpu",
    "detect_nvdec_support",
    "estimate_optimal_batch_size",
    "get_decode_info",
    "get_resolution_batch_size_cap",
    "is_codec_nvdec_compatible",
    "select_operation_processor",
    "should_use_async_io",
]

# GPU detection functions are available via detection_gpu module
# Import them lazily to avoid CuPy import errors when GPU is not available
try:
    from .detection_gpu import (
        compute_histogram_distance_batch_gpu,
        compute_histogram_distance_gpu,
        compute_pixel_difference_batch_gpu,
        compute_pixel_difference_gpu,
        free_gpu_memory,
    )

    __all__.extend(
        [
            "compute_histogram_distance_batch_gpu",
            "compute_histogram_distance_gpu",
            "compute_pixel_difference_batch_gpu",
            "compute_pixel_difference_gpu",
            "free_gpu_memory",
        ]
    )
except ImportError:
    # CuPy not available, GPU functions not exported
    pass
