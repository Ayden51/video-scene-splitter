"""
GPU utilities for hardware acceleration detection and configuration.

This module provides GPU detection, validation, and configuration functionality
with a pluggable backend architecture to support multiple acceleration frameworks
(CuPy now, PyTorch in future versions).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class ProcessorType(Enum):
    """Processing mode for video scene detection."""

    CPU = "cpu"
    GPU = "gpu"
    AUTO = "auto"

    @classmethod
    def from_string(cls, value: str) -> ProcessorType:
        """Convert a string to ProcessorType.

        Args:
            value: String value ("cpu", "gpu", or "auto").

        Returns:
            Corresponding ProcessorType enum value.

        Raises:
            ValueError: If value is not a valid processor type.
        """
        value_lower = value.lower()
        for member in cls:
            if member.value == value_lower:
                return member
        valid_values = [m.value for m in cls]
        raise ValueError(f"Invalid processor type: {value}. Must be one of {valid_values}")


@dataclass
class GPUInfo:
    """Information about available GPU hardware.

    Attributes:
        available: Whether a compatible GPU is available.
        name: GPU device name (e.g., "NVIDIA GeForce RTX 4090").
        cuda_version: CUDA runtime version string.
        driver_version: NVIDIA driver version string.
        memory_total_mb: Total GPU memory in megabytes.
        memory_free_mb: Available GPU memory in megabytes.
        compute_capability: CUDA compute capability tuple (major, minor).
        backend: Name of the acceleration backend (e.g., "cupy", "pytorch").
    """

    available: bool
    name: str | None = None
    cuda_version: str | None = None
    driver_version: str | None = None
    memory_total_mb: int | None = None
    memory_free_mb: int | None = None
    compute_capability: tuple[int, int] | None = None
    backend: str | None = None


@dataclass
class GPUConfig:
    """Configuration for GPU processing.

    Attributes:
        batch_size: Number of frames to process in a single GPU batch.
        memory_fraction: Maximum fraction of GPU memory to use (0.0-1.0).
        use_nvenc: Whether to use NVIDIA NVENC for video encoding.
    """

    batch_size: int = 30
    memory_fraction: float = 0.8
    use_nvenc: bool = True


class AccelerationBackend(ABC):
    """Abstract base class for GPU acceleration backends.

    This interface allows pluggable acceleration frameworks (CuPy, PyTorch, etc.)
    to be used interchangeably for GPU-accelerated operations.

    Implementations must provide GPU detection and array conversion capabilities.
    Detection-specific operations are handled by separate detection backend classes.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the backend name (e.g., 'cupy', 'pytorch')."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available and functional."""

    @abstractmethod
    def get_gpu_info(self) -> GPUInfo:
        """Detect and return GPU hardware information."""

    @abstractmethod
    def configure_memory(self, fraction: float) -> None:
        """Configure GPU memory usage limits.

        Args:
            fraction: Maximum fraction of GPU memory to use (0.0-1.0).
        """

    @abstractmethod
    def to_gpu(self, array):
        """Transfer a NumPy array to GPU memory.

        Args:
            array: NumPy array to transfer.

        Returns:
            GPU array in the backend's native format.
        """

    @abstractmethod
    def to_cpu(self, array):
        """Transfer a GPU array back to CPU memory.

        Args:
            array: GPU array in the backend's native format.

        Returns:
            NumPy array.
        """

    @abstractmethod
    def free_memory(self) -> None:
        """Release unused GPU memory back to the system."""


class CuPyBackend(AccelerationBackend):
    """CuPy-based GPU acceleration backend.

    Uses CuPy (CUDA-accelerated NumPy) for GPU operations.
    Optimized for CUDA 13.0+ on Windows 11.
    """

    _instance: CuPyBackend | None = None
    _cp = None  # Lazy-loaded CuPy module

    def __new__(cls):
        """Singleton pattern for backend instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def name(self) -> str:
        return "cupy"

    def _ensure_cupy(self) -> bool:
        """Lazily import CuPy and cache the module."""
        if self._cp is not None:
            return True
        try:
            import cupy as cp

            CuPyBackend._cp = cp
            return True
        except ImportError:
            return False

    def is_available(self) -> bool:
        """Check if CuPy and CUDA are available."""
        if not self._ensure_cupy():
            return False
        try:
            # Try to access the first CUDA device
            _ = self._cp.cuda.Device(0).compute_capability
            return True
        except Exception:
            return False

    def get_gpu_info(self) -> GPUInfo:
        """Detect GPU using CuPy CUDA runtime."""
        if not self.is_available():
            return GPUInfo(available=False, backend=self.name)

        try:
            cp = self._cp
            device = cp.cuda.Device(0)

            # Get device properties via CUDA runtime API
            props = cp.cuda.runtime.getDeviceProperties(0)

            # Get device name (returns bytes, decode to string)
            name_raw = props.get("name")
            name = name_raw.decode("utf-8") if isinstance(name_raw, bytes) else name_raw

            # Get compute capability from device
            compute_cap = device.compute_capability

            # Get memory info (free, total) in bytes
            mem_info = device.mem_info
            memory_free_mb = mem_info[0] // (1024 * 1024)
            memory_total_mb = mem_info[1] // (1024 * 1024)

            # Get CUDA runtime version (returns int like 13000 for CUDA 13.0)
            runtime_ver = cp.cuda.runtime.runtimeGetVersion()
            cuda_version = f"{runtime_ver // 1000}.{(runtime_ver % 1000) // 10}"

            # Get driver version if available
            driver_version = None
            try:
                driver_ver = cp.cuda.runtime.driverGetVersion()
                driver_version = f"{driver_ver // 1000}.{(driver_ver % 1000) // 10}"
            except Exception:
                pass

            return GPUInfo(
                available=True,
                name=name,
                cuda_version=cuda_version,
                driver_version=driver_version,
                memory_total_mb=memory_total_mb,
                memory_free_mb=memory_free_mb,
                compute_capability=compute_cap,
                backend=self.name,
            )
        except Exception:
            return GPUInfo(available=False, backend=self.name)

    def configure_memory(self, fraction: float) -> None:
        """Configure CuPy memory pool limits."""
        if not self.is_available():
            return

        cp = self._cp
        mempool = cp.get_default_memory_pool()

        # Get total memory and set limit
        device = cp.cuda.Device(0)
        total_mem = device.mem_info[1]
        mempool.set_limit(size=int(total_mem * fraction))

    def to_gpu(self, array):
        """Transfer NumPy array to CuPy GPU array."""
        if not self._ensure_cupy():
            raise RuntimeError("CuPy is not available")
        return self._cp.asarray(array)

    def to_cpu(self, array):
        """Transfer CuPy GPU array to NumPy array."""
        if not self._ensure_cupy():
            raise RuntimeError("CuPy is not available")
        return self._cp.asnumpy(array)

    def free_memory(self) -> None:
        """Release unused GPU memory from CuPy memory pool."""
        if not self.is_available():
            return

        cp = self._cp
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()


# =============================================================================
# Backend Registry and Selection
# =============================================================================

# Registry of available acceleration backends
_BACKENDS: dict[str, type[AccelerationBackend]] = {
    "cupy": CuPyBackend,
    # Future: "pytorch": PyTorchBackend,
}

_active_backend: AccelerationBackend | None = None


def get_available_backends() -> list[str]:
    """Return list of available (installed and functional) backend names."""
    available = []
    for name, backend_class in _BACKENDS.items():
        try:
            backend = backend_class()
            if backend.is_available():
                available.append(name)
        except Exception:
            pass
    return available


def get_backend(name: str | None = None) -> AccelerationBackend | None:
    """Get a specific backend instance or the first available one.

    Args:
        name: Backend name (e.g., 'cupy'). If None, returns the first available.

    Returns:
        AccelerationBackend instance or None if not available.
    """
    global _active_backend

    if name is not None:
        if name not in _BACKENDS:
            return None
        backend = _BACKENDS[name]()
        if backend.is_available():
            _active_backend = backend
            return backend
        return None

    # Auto-select first available backend
    for backend_name in _BACKENDS:
        backend = _BACKENDS[backend_name]()
        if backend.is_available():
            _active_backend = backend
            return backend

    return None


def detect_cuda_gpu() -> GPUInfo:
    """Detect CUDA-capable GPU using available backends.

    Tries each registered backend until one successfully detects a GPU.

    Returns:
        GPUInfo: GPU information or unavailable status.
    """
    for backend_class in _BACKENDS.values():
        try:
            backend = backend_class()
            info = backend.get_gpu_info()
            if info.available:
                return info
        except Exception:
            pass

    return GPUInfo(available=False)


def select_processor(requested: ProcessorType, gpu_info: GPUInfo | None = None) -> ProcessorType:
    """Select actual processor based on request and GPU availability.

    Args:
        requested: User-requested processor type.
        gpu_info: Pre-detected GPU info (optional, will detect if not provided).

    Returns:
        ProcessorType: The actual processor that will be used.

    Raises:
        RuntimeError: If GPU is requested but not available.
    """
    if requested == ProcessorType.CPU:
        return ProcessorType.CPU

    if gpu_info is None:
        gpu_info = detect_cuda_gpu()

    if requested == ProcessorType.GPU:
        if not gpu_info.available:
            raise RuntimeError(
                "GPU processing requested but no CUDA GPU available.\n"
                "To install GPU support with uv:\n"
                "  CUDA 13.x: uv sync --extra gpu\n"
                "  CUDA 12.x: uv sync --extra gpu-cuda12\n\n"
                "Alternative with pip:\n"
                "  CUDA 13.x: pip install cupy-cuda13x\n"
                "  CUDA 12.x: pip install cupy-cuda12x"
            )
        return ProcessorType.GPU

    # AUTO mode: use GPU if available, fallback to CPU
    if gpu_info.available:
        return ProcessorType.GPU
    return ProcessorType.CPU


def estimate_optimal_batch_size(
    frame_height: int,
    frame_width: int,
    available_memory_mb: int,
    safety_factor: float = 0.7,
) -> int:
    """
    Calculate optimal batch size based on GPU memory and frame dimensions.

    Memory per frame estimate:
    - BGR frame: H * W * 3 bytes
    - Grayscale: H * W bytes (for pixel difference)
    - HSV: H * W * 3 bytes (for histogram)
    - Histograms: ~170 bins * 4 bytes * 3 channels = ~2KB
    - Working memory: ~2x frame size for intermediate results

    Total per frame: ~2.5 * (H * W * 3) bytes

    Args:
        frame_height: Height of video frames in pixels.
        frame_width: Width of video frames in pixels.
        available_memory_mb: Available GPU memory in megabytes.
        safety_factor: Fraction of available memory to use (0.0-1.0).
            Default is 0.7 to leave headroom for CuPy overhead.

    Returns:
        int: Recommended batch size (clamped to 5-120 range).
    """
    # Memory per frame (all intermediate buffers)
    # BGR frame + grayscale + HSV + working buffers
    bytes_per_frame = int(2.5 * frame_height * frame_width * 3)

    # Available bytes with safety factor
    available_bytes = available_memory_mb * 1024 * 1024 * safety_factor

    # Calculate optimal batch
    optimal_batch = int(available_bytes / bytes_per_frame) if bytes_per_frame > 0 else 30

    # Clamp to reasonable range (5-120 frames)
    return max(5, min(optimal_batch, 120))


@dataclass
class AutoModeConfig:
    """Configuration for AUTO mode hybrid processing.

    Based on actual benchmark results (2026-01-09):
    - SD (480p): GPU provides 1.32x overall speedup
    - HD (1080p): GPU provides 1.39x overall speedup (best benefit)
    - 4K (2160p): GPU provides 0.88x speedup (CPU faster due to transfer overhead)

    AUTO mode optimizes performance by using the best processor for each operation:
    - GPU for pixel difference on SD/HD content (1.32-1.74x speedup)
    - CPU for pixel difference on 4K content (GPU 0.88x is slower)
    - CPU for histogram computation (always, as CPU is faster than GPU)
    - Async I/O when GPU processing time >= 10ms per batch (1.01-1.54x speedup)

    Attributes:
        min_resolution_for_gpu: Minimum vertical resolution for GPU processing.
            Content below this uses CPU only. Default: 480 (SD threshold).
        max_resolution_for_gpu: Maximum vertical resolution for GPU pixel diff.
            Content above this uses CPU (GPU overhead exceeds benefit). Default: 1440 (below 4K).
        use_gpu_for_pixel_diff: Whether to use GPU for pixel difference on SD/HD.
            Default: True (GPU is 1.32-1.74x faster for SD/HD content).
        use_gpu_for_histogram: Whether to use GPU for histogram computation.
            Default: False (CPU is faster due to transfer overhead).
        use_async_io: Whether to use async I/O for GPU processing.
            Default: True (1.01-1.54x speedup when GPU processing time >= 10ms).
        min_gpu_delay_for_async_ms: Minimum GPU processing time per batch to benefit from async.
            Default: 10ms (async provides 1.01-1.54x speedup above this threshold).
        max_batch_size_sd: Maximum batch size for SD content (<720p).
        max_batch_size_hd: Maximum batch size for HD content (720p-1080p).
        max_batch_size_4k: Maximum batch size for 4K content (≥2160p).
            Note: 4K uses CPU for scene detection, but batch size still limits memory.
        memory_fraction: Conservative memory usage fraction. Default: 0.7.
    """

    # Resolution thresholds based on benchmark results
    min_resolution_for_gpu: int = 480  # SD threshold (GPU provides 1.32x speedup)
    max_resolution_for_gpu: int = 1440  # Below 4K (4K GPU is 0.88x, slower than CPU)
    use_gpu_for_pixel_diff: bool = True  # GPU 1.32-1.74x faster for SD/HD
    use_gpu_for_histogram: bool = False  # CPU always faster (transfer overhead)

    # Async I/O settings based on benchmark results
    use_async_io: bool = True  # Async provides 1.01-1.54x speedup with GPU
    min_gpu_delay_for_async_ms: float = 10.0  # Minimum GPU time to benefit from async

    # Batch size caps (prevent OOM, especially for 4K even in CPU mode)
    max_batch_size_sd: int = 60  # SD: minimal GPU benefit, larger batches OK
    max_batch_size_hd: int = 30  # HD: balanced (best GPU benefit)
    max_batch_size_4k: int = 15  # 4K: memory constrained, CPU mode
    memory_fraction: float = 0.7  # Conservative memory usage


def get_resolution_batch_size_cap(
    frame_height: int,
    config: AutoModeConfig | None = None,
) -> int:
    """Get the maximum batch size based on video resolution.

    Higher resolution content requires smaller batches to avoid OOM errors.

    Args:
        frame_height: Height of video frames in pixels.
        config: Optional AutoModeConfig with custom batch caps.

    Returns:
        Maximum recommended batch size for the resolution.
    """
    if config is None:
        config = AutoModeConfig()

    if frame_height >= 2160:  # 4K
        return config.max_batch_size_4k
    elif frame_height >= 720:  # HD
        return config.max_batch_size_hd
    else:  # SD
        return config.max_batch_size_sd


def select_operation_processor(
    operation: str,
    frame_height: int,
    frame_width: int,
    gpu_info: GPUInfo,
    config: AutoModeConfig | None = None,
    verbose: bool = False,
) -> ProcessorType:
    """Select the optimal processor for a specific operation in AUTO mode.

    Based on actual benchmark results (2026-01-09):
    - SD (480p): GPU pixel diff 1.36-1.74x faster, overall 1.32x speedup
    - HD (1080p): GPU pixel diff 0.98-2.04x faster, overall 1.39x speedup
    - 4K (2160p): GPU pixel diff 0.57-1.46x, overall 0.88x (CPU faster)
    - Histogram: CPU always faster (transfer overhead)

    Args:
        operation: Operation type ("pixel_diff" or "histogram").
        frame_height: Height of video frames in pixels.
        frame_width: Width of video frames in pixels (for memory estimation).
        gpu_info: GPU detection information.
        config: Optional AutoModeConfig for customization.
        verbose: If True, log decision rationale.

    Returns:
        ProcessorType.GPU or ProcessorType.CPU based on optimal selection.
    """
    if config is None:
        config = AutoModeConfig()

    # Classify resolution for decision logging
    if frame_height >= 2160:
        resolution_class = "4K"
    elif frame_height >= 720:
        resolution_class = "HD"
    else:
        resolution_class = "SD"

    # No GPU available - always CPU
    if not gpu_info.available:
        if verbose:
            print(f"    → {operation}: CPU (GPU not available)")
        return ProcessorType.CPU

    # Histogram: always CPU (transfer overhead makes GPU slower)
    if operation == "histogram":
        if config.use_gpu_for_histogram:
            if verbose:
                print("    → histogram: GPU (forced via config)")
            return ProcessorType.GPU
        if verbose:
            print("    → histogram: CPU (1.1-1.7x faster due to transfer overhead)")
        return ProcessorType.CPU

    # Pixel difference: resolution-based selection
    if operation == "pixel_diff":
        if not config.use_gpu_for_pixel_diff:
            if verbose:
                print("    → pixel_diff: CPU (forced via config)")
            return ProcessorType.CPU

        # 4K content: CPU (GPU 0.88x speedup means CPU is faster)
        if frame_height >= 2160:
            if verbose:
                print(
                    f"    → pixel_diff: CPU for {resolution_class} "
                    f"(benchmark: GPU 0.88x speedup, CPU faster)"
                )
            return ProcessorType.CPU

        # HD content: GPU (1.39x overall speedup)
        if frame_height >= 720:
            if verbose:
                print(
                    f"    → pixel_diff: GPU for {resolution_class} "
                    f"(benchmark: 1.39x overall speedup)"
                )
            return ProcessorType.GPU

        # SD content: GPU (1.32x overall speedup)
        if frame_height >= config.min_resolution_for_gpu:
            if verbose:
                print(
                    f"    → pixel_diff: GPU for {resolution_class} "
                    f"(benchmark: 1.32x overall speedup)"
                )
            return ProcessorType.GPU

        # Below minimum resolution: CPU
        if verbose:
            print(
                f"    → pixel_diff: CPU for {resolution_class} "
                f"(below {config.min_resolution_for_gpu}p threshold)"
            )
        return ProcessorType.CPU

    # Unknown operation - default to CPU
    if verbose:
        print(f"    → {operation}: CPU (unknown operation)")
    return ProcessorType.CPU


def should_use_async_io(
    processor: ProcessorType,
    gpu_info: GPUInfo,
    config: AutoModeConfig | None = None,
    verbose: bool = False,
) -> bool:
    """Determine whether to use async I/O for frame reading.

    Based on benchmark results (2026-01-09):
    - Without GPU delay (0ms): Async has ~10-20% overhead (0.76-0.93x)
    - With GPU delay >= 10ms: Async provides 1.01-1.54x speedup

    Async I/O benefits come from overlapping disk I/O with GPU computation.
    For CPU-only processing, sync I/O is preferred (no overhead).

    Args:
        processor: The processor being used for scene detection.
        gpu_info: GPU detection information.
        config: Optional AutoModeConfig for customization.
        verbose: If True, log decision rationale.

    Returns:
        True if async I/O should be used, False for sync I/O.
    """
    if config is None:
        config = AutoModeConfig()

    # CPU mode: use sync I/O (no benefit from async, adds overhead)
    if processor == ProcessorType.CPU:
        if verbose:
            print("    → Frame I/O: Sync (CPU mode, no async benefit)")
        return False

    # GPU not available: use sync I/O
    if not gpu_info.available:
        if verbose:
            print("    → Frame I/O: Sync (GPU not available)")
        return False

    # Async disabled in config
    if not config.use_async_io:
        if verbose:
            print("    → Frame I/O: Sync (disabled via config)")
        return False

    # GPU mode: use async I/O (1.01-1.54x speedup)
    if verbose:
        print(
            f"    → Frame I/O: Async (GPU processing >= {config.min_gpu_delay_for_async_ms}ms, "
            f"1.01-1.54x speedup)"
        )
    return True


def print_gpu_status(
    gpu_info: GPUInfo, selected_processor: ProcessorType, debug: bool = False
) -> None:
    """Print user-friendly GPU detection status.

    Args:
        gpu_info: Detected GPU information.
        selected_processor: The processor that will be used.
        debug: If True, show detailed GPU information. If False, show brief status.
    """
    if debug:
        # Detailed GPU information for debug mode
        print("\n" + "=" * 70)
        print("🔍 Hardware Detection & Configuration")
        print("=" * 70)

        if gpu_info.available:
            print("GPU Status:        ✓ Available")
            print(f"GPU Name:          {gpu_info.name or 'NVIDIA GPU'}")
            if gpu_info.memory_total_mb:
                print(f"GPU Memory:        {gpu_info.memory_total_mb}MB total", end="")
                if gpu_info.memory_free_mb:
                    print(f", {gpu_info.memory_free_mb}MB free")
                else:
                    print()
            if gpu_info.cuda_version:
                print(f"CUDA Version:      {gpu_info.cuda_version}")
            if gpu_info.driver_version:
                print(f"Driver Version:    {gpu_info.driver_version}")
            if gpu_info.compute_capability:
                print(f"Compute Capability: {gpu_info.compute_capability}")
            if gpu_info.backend:
                print(f"Backend:           {gpu_info.backend}")

            print(f"\nProcessor Mode:    {selected_processor.value.upper()}")
            if selected_processor == ProcessorType.GPU:
                print("Acceleration:      ✓ GPU acceleration enabled")
            else:
                print("Acceleration:      CPU mode (GPU available but not selected)")
        else:
            print("GPU Status:        ✗ Not available")
            print(f"Processor Mode:    {selected_processor.value.upper()}")
            print("Acceleration:      CPU processing only")
            print("\n💡 To enable GPU acceleration:")
            print("   With uv:")
            print("     CUDA 13.x: uv sync --extra gpu")
            print("     CUDA 12.x: uv sync --extra gpu-cuda12")
            print("   With pip:")
            print("     CUDA 13.x: pip install cupy-cuda13x")
            print("     CUDA 12.x: pip install cupy-cuda12x")

        print("=" * 70 + "\n")
    else:
        # Brief status for normal mode
        if gpu_info.available and selected_processor == ProcessorType.GPU:
            print(f"Using GPU acceleration ({gpu_info.name or 'NVIDIA GPU'})")
        else:
            print("Using CPU processing")
