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
        use_nvdec: Whether to use NVIDIA NVDEC for video decoding.
        use_pyav_hw: Whether to use PyAV hardware acceleration.
    """

    batch_size: int = 30
    memory_fraction: float = 0.8
    use_nvenc: bool = True
    use_nvdec: bool = True
    use_pyav_hw: bool = True


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
