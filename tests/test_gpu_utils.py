"""
Unit tests for video_scene_splitter.gpu_utils module.

Tests the GPU detection, configuration, and backend selection functionality.
All tests use mocking to work on systems without GPU hardware.
"""

import pytest

from video_scene_splitter.gpu_utils import (
    AccelerationBackend,
    CuPyBackend,
    GPUConfig,
    GPUInfo,
    ProcessorType,
    detect_cuda_gpu,
    estimate_optimal_batch_size,
    get_available_backends,
    get_backend,
    print_gpu_status,
    select_processor,
)


@pytest.fixture(autouse=True)
def reset_cupy_backend_singleton():
    """Reset CuPyBackend singleton state before and after each test.

    This ensures tests don't interfere with each other through the singleton.
    """
    # Store original state
    original_instance = CuPyBackend._instance
    original_cp = CuPyBackend._cp

    yield

    # Restore original state after test
    CuPyBackend._instance = original_instance
    CuPyBackend._cp = original_cp


class TestProcessorType:
    """Tests for ProcessorType enum."""

    def test_cpu_value(self):
        """CPU processor type should have value 'cpu'."""
        assert ProcessorType.CPU.value == "cpu"

    def test_gpu_value(self):
        """GPU processor type should have value 'gpu'."""
        assert ProcessorType.GPU.value == "gpu"

    def test_auto_value(self):
        """AUTO processor type should have value 'auto'."""
        assert ProcessorType.AUTO.value == "auto"

    def test_from_string(self):
        """ProcessorType should be constructible from string values."""
        assert ProcessorType("cpu") == ProcessorType.CPU
        assert ProcessorType("gpu") == ProcessorType.GPU
        assert ProcessorType("auto") == ProcessorType.AUTO


class TestGPUInfo:
    """Tests for GPUInfo dataclass."""

    def test_unavailable_gpu_info(self):
        """GPUInfo with available=False should have None for optional fields."""
        info = GPUInfo(available=False)
        assert info.available is False
        assert info.name is None
        assert info.cuda_version is None
        assert info.memory_total_mb is None

    def test_available_gpu_info(self):
        """GPUInfo should store all GPU properties correctly."""
        info = GPUInfo(
            available=True,
            name="Test GPU",
            cuda_version="13.0",
            driver_version="550.0",
            memory_total_mb=8192,
            memory_free_mb=6144,
            compute_capability=(8, 9),
            backend="cupy",
        )
        assert info.available is True
        assert info.name == "Test GPU"
        assert info.cuda_version == "13.0"
        assert info.memory_total_mb == 8192
        assert info.compute_capability == (8, 9)
        assert info.backend == "cupy"


class TestGPUConfig:
    """Tests for GPUConfig dataclass."""

    def test_default_values(self):
        """GPUConfig should have sensible defaults."""
        config = GPUConfig()
        assert config.batch_size == 30
        assert config.memory_fraction == 0.8
        assert config.use_nvenc is True
        assert config.use_nvdec is True
        assert config.use_pyav_hw is True

    def test_custom_values(self):
        """GPUConfig should accept custom values."""
        config = GPUConfig(batch_size=60, memory_fraction=0.5, use_nvenc=False)
        assert config.batch_size == 60
        assert config.memory_fraction == 0.5
        assert config.use_nvenc is False


class TestCuPyBackend:
    """Tests for CuPyBackend class."""

    def test_backend_name(self):
        """CuPyBackend should return 'cupy' as name."""
        backend = CuPyBackend()
        assert backend.name == "cupy"

    def test_singleton_pattern(self):
        """CuPyBackend should use singleton pattern."""
        backend1 = CuPyBackend()
        backend2 = CuPyBackend()
        assert backend1 is backend2

    def test_is_abstract_backend(self):
        """CuPyBackend should inherit from AccelerationBackend."""
        backend = CuPyBackend()
        assert isinstance(backend, AccelerationBackend)

    def test_is_available_when_cupy_not_installed(self, mocker):
        """is_available should return False when CuPy is not installed."""
        # Reset singleton state
        CuPyBackend._instance = None
        CuPyBackend._cp = None

        # Mock the import to fail
        mocker.patch.dict("sys.modules", {"cupy": None})
        backend = CuPyBackend()

        # Force reimport attempt by resetting _cp
        backend._cp = None
        result = backend.is_available()
        assert result is False

    def test_get_gpu_info_unavailable(self, mocker):
        """get_gpu_info should return unavailable when CuPy not installed."""
        # Reset singleton state
        CuPyBackend._instance = None
        CuPyBackend._cp = None

        mocker.patch.dict("sys.modules", {"cupy": None})
        backend = CuPyBackend()
        backend._cp = None

        info = backend.get_gpu_info()
        assert info.available is False
        assert info.backend == "cupy"

    def test_configure_memory_when_unavailable(self, mocker):
        """configure_memory should do nothing when GPU unavailable."""
        backend = CuPyBackend()
        mocker.patch.object(backend, "is_available", return_value=False)

        # Should not raise
        backend.configure_memory(0.5)

    def test_free_memory_when_unavailable(self, mocker):
        """free_memory should do nothing when GPU unavailable."""
        backend = CuPyBackend()
        mocker.patch.object(backend, "is_available", return_value=False)

        # Should not raise
        backend.free_memory()

    def test_to_gpu_raises_when_unavailable(self, mocker):
        """to_gpu should raise RuntimeError when CuPy unavailable."""
        import numpy as np

        # Reset singleton state
        CuPyBackend._instance = None
        CuPyBackend._cp = None

        backend = CuPyBackend()
        backend._cp = None  # Ensure CuPy is not loaded
        mocker.patch.object(backend, "_ensure_cupy", return_value=False)

        with pytest.raises(RuntimeError, match="CuPy is not available"):
            backend.to_gpu(np.array([1, 2, 3]))

    def test_to_cpu_raises_when_unavailable(self, mocker):
        """to_cpu should raise RuntimeError when CuPy unavailable."""
        # Reset singleton state
        CuPyBackend._instance = None
        CuPyBackend._cp = None

        backend = CuPyBackend()
        backend._cp = None
        mocker.patch.object(backend, "_ensure_cupy", return_value=False)

        with pytest.raises(RuntimeError, match="CuPy is not available"):
            backend.to_cpu(None)

    def test_ensure_cupy_returns_true_when_already_loaded(self, mocker):
        """_ensure_cupy should return True immediately if _cp is set."""
        backend = CuPyBackend()

        # Mock _cp as if CuPy was loaded
        mock_cp = mocker.MagicMock()
        backend._cp = mock_cp

        result = backend._ensure_cupy()
        assert result is True

    def test_is_available_catches_exceptions(self, mocker):
        """is_available should return False when CUDA device access fails."""
        backend = CuPyBackend()

        # Mock CuPy module
        mock_cp = mocker.MagicMock()
        mock_cp.cuda.Device.side_effect = Exception("CUDA error")
        backend._cp = mock_cp

        # Should catch the exception and return False
        mocker.patch.object(backend, "_ensure_cupy", return_value=True)
        result = backend.is_available()
        assert result is False

    def test_get_gpu_info_with_mocked_cupy(self, mocker):
        """get_gpu_info should return valid GPUInfo when CuPy is available."""
        backend = CuPyBackend()

        # Create mock CuPy module
        mock_device = mocker.MagicMock()
        mock_device.compute_capability = (8, 6)
        mock_device.mem_info = (8 * 1024 * 1024 * 1024, 10 * 1024 * 1024 * 1024)

        mock_cp = mocker.MagicMock()
        mock_cp.cuda.Device.return_value = mock_device
        # getDeviceProperties returns dict with 'name' as bytes
        mock_cp.cuda.runtime.getDeviceProperties.return_value = {"name": b"NVIDIA GeForce RTX 3080"}
        # runtimeGetVersion returns int like 12000 for CUDA 12.0
        mock_cp.cuda.runtime.runtimeGetVersion.return_value = 12000
        mock_cp.cuda.runtime.driverGetVersion.return_value = 12010

        backend._cp = mock_cp
        mocker.patch.object(backend, "is_available", return_value=True)

        info = backend.get_gpu_info()
        assert info.available is True
        assert info.name == "NVIDIA GeForce RTX 3080"
        assert info.compute_capability == (8, 6)
        assert info.memory_total_mb == 10 * 1024
        assert info.memory_free_mb == 8 * 1024
        assert info.backend == "cupy"

    def test_get_gpu_info_driver_version_exception(self, mocker):
        """get_gpu_info should handle driver version exception gracefully."""
        backend = CuPyBackend()

        mock_device = mocker.MagicMock()
        mock_device.compute_capability = (7, 5)
        mock_device.mem_info = (4 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024)

        mock_cp = mocker.MagicMock()
        mock_cp.cuda.Device.return_value = mock_device
        mock_cp.cuda.runtime.getDeviceProperties.return_value = {"name": b"Test GPU"}
        # runtimeGetVersion returns int like 11080 for CUDA 11.8
        mock_cp.cuda.runtime.runtimeGetVersion.return_value = 11080
        mock_cp.cuda.runtime.driverGetVersion.side_effect = Exception("Driver error")

        backend._cp = mock_cp
        mocker.patch.object(backend, "is_available", return_value=True)

        info = backend.get_gpu_info()
        assert info.available is True
        assert info.driver_version is None

    def test_get_gpu_info_exception_returns_unavailable(self, mocker):
        """get_gpu_info should return unavailable on exception."""
        backend = CuPyBackend()

        mock_cp = mocker.MagicMock()
        mock_cp.cuda.Device.side_effect = Exception("CUDA error")

        backend._cp = mock_cp
        mocker.patch.object(backend, "is_available", return_value=True)

        info = backend.get_gpu_info()
        assert info.available is False

    def test_configure_memory_with_mocked_cupy(self, mocker):
        """configure_memory should set memory pool limit."""
        backend = CuPyBackend()

        mock_mempool = mocker.MagicMock()
        mock_device = mocker.MagicMock()
        mock_device.mem_info = (4 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024)

        mock_cp = mocker.MagicMock()
        mock_cp.get_default_memory_pool.return_value = mock_mempool
        mock_cp.cuda.Device.return_value = mock_device

        backend._cp = mock_cp
        mocker.patch.object(backend, "is_available", return_value=True)

        backend.configure_memory(0.5)

        # Verify memory pool limit was set to 50% of total memory
        expected_limit = int(8 * 1024 * 1024 * 1024 * 0.5)
        mock_mempool.set_limit.assert_called_once_with(size=expected_limit)

    def test_free_memory_with_mocked_cupy(self, mocker):
        """free_memory should free all memory pool blocks."""
        backend = CuPyBackend()

        mock_mempool = mocker.MagicMock()
        mock_pinned_mempool = mocker.MagicMock()

        mock_cp = mocker.MagicMock()
        mock_cp.get_default_memory_pool.return_value = mock_mempool
        mock_cp.get_default_pinned_memory_pool.return_value = mock_pinned_mempool

        backend._cp = mock_cp
        mocker.patch.object(backend, "is_available", return_value=True)

        backend.free_memory()

        mock_mempool.free_all_blocks.assert_called_once()
        mock_pinned_mempool.free_all_blocks.assert_called_once()

    def test_to_gpu_with_mocked_cupy(self, mocker):
        """to_gpu should convert numpy array to cupy array."""
        import numpy as np

        backend = CuPyBackend()

        mock_gpu_array = mocker.MagicMock()
        mock_cp = mocker.MagicMock()
        mock_cp.asarray.return_value = mock_gpu_array

        backend._cp = mock_cp
        mocker.patch.object(backend, "_ensure_cupy", return_value=True)

        test_array = np.array([1, 2, 3])
        result = backend.to_gpu(test_array)

        mock_cp.asarray.assert_called_once_with(test_array)
        assert result is mock_gpu_array

    def test_to_cpu_with_mocked_cupy(self, mocker):
        """to_cpu should convert cupy array to numpy array."""
        import numpy as np

        backend = CuPyBackend()

        expected_array = np.array([1, 2, 3])
        mock_cp = mocker.MagicMock()
        mock_cp.asnumpy.return_value = expected_array

        backend._cp = mock_cp
        mocker.patch.object(backend, "_ensure_cupy", return_value=True)

        mock_gpu_array = mocker.MagicMock()
        result = backend.to_cpu(mock_gpu_array)

        mock_cp.asnumpy.assert_called_once_with(mock_gpu_array)
        assert result is expected_array


class TestEstimateOptimalBatchSize:
    """Tests for estimate_optimal_batch_size function."""

    def test_returns_integer(self):
        """estimate_optimal_batch_size should return an integer."""
        result = estimate_optimal_batch_size(
            frame_height=1080, frame_width=1920, available_memory_mb=8192
        )
        assert isinstance(result, int)

    def test_minimum_batch_size_is_5(self):
        """Batch size should never be less than 5."""
        # Very low memory should still return at least 5
        result = estimate_optimal_batch_size(
            frame_height=2160,
            frame_width=3840,
            available_memory_mb=1,  # Unrealistically low
        )
        assert result >= 5

    def test_maximum_batch_size_is_120(self):
        """Batch size should never exceed 120."""
        # Very high memory should still cap at 120
        result = estimate_optimal_batch_size(
            frame_height=100,
            frame_width=100,
            available_memory_mb=1000000,  # Unrealistically high
        )
        assert result <= 120

    def test_larger_frames_reduce_batch_size(self):
        """Larger frames should result in smaller batch sizes."""
        result_sd = estimate_optimal_batch_size(
            frame_height=480, frame_width=640, available_memory_mb=8192
        )
        result_4k = estimate_optimal_batch_size(
            frame_height=2160, frame_width=3840, available_memory_mb=8192
        )
        # 4K frames are much larger, so batch size should be smaller
        assert result_4k < result_sd

    def test_more_memory_increases_batch_size(self):
        """More GPU memory should allow larger batch sizes."""
        result_4gb = estimate_optimal_batch_size(
            frame_height=1080, frame_width=1920, available_memory_mb=4096
        )
        result_16gb = estimate_optimal_batch_size(
            frame_height=1080, frame_width=1920, available_memory_mb=16384
        )
        # More memory should allow larger batches
        assert result_16gb >= result_4gb

    def test_safety_factor_reduces_effective_memory(self):
        """Safety factor should reduce effective memory available."""
        result_high_safety = estimate_optimal_batch_size(
            frame_height=1080, frame_width=1920, available_memory_mb=8192, safety_factor=0.3
        )
        result_low_safety = estimate_optimal_batch_size(
            frame_height=1080, frame_width=1920, available_memory_mb=8192, safety_factor=0.9
        )
        # Lower safety factor means more usable memory, larger batch
        assert result_low_safety >= result_high_safety

    def test_typical_hd_video_with_8gb(self):
        """Typical HD video with 8GB VRAM should return reasonable batch size."""
        result = estimate_optimal_batch_size(
            frame_height=1080, frame_width=1920, available_memory_mb=8192
        )
        # Should be in a reasonable range for 8GB GPU
        assert 15 <= result <= 120

    def test_zero_dimensions_returns_default(self):
        """Zero dimensions should return default batch size (30)."""
        result = estimate_optimal_batch_size(
            frame_height=0, frame_width=0, available_memory_mb=8192
        )
        assert result == 30


class TestSelectProcessor:
    """Tests for select_processor function."""

    def test_gpu_raises_when_unavailable(self):
        """Requesting GPU should raise error when unavailable."""
        with pytest.raises(RuntimeError, match="GPU processing requested"):
            select_processor(ProcessorType.GPU, GPUInfo(available=False))

    def test_auto_returns_gpu_when_available(self):
        """AUTO mode should return GPU when available."""
        result = select_processor(ProcessorType.AUTO, GPUInfo(available=True))
        assert result == ProcessorType.GPU

    def test_auto_returns_cpu_when_unavailable(self):
        """AUTO mode should fallback to CPU when GPU unavailable."""
        result = select_processor(ProcessorType.AUTO, GPUInfo(available=False))
        assert result == ProcessorType.CPU

    def test_cpu_always_returns_cpu(self):
        """CPU mode should always select CPU regardless of GPU availability."""
        result_when_gpu_available = select_processor(
            ProcessorType.CPU,
            GPUInfo(available=True),
        )
        result_when_gpu_unavailable = select_processor(
            ProcessorType.CPU,
            GPUInfo(available=False),
        )

        assert result_when_gpu_available == ProcessorType.CPU
        assert result_when_gpu_unavailable == ProcessorType.CPU


class TestDetectCudaGpu:
    """Tests for detect_cuda_gpu function."""

    def test_returns_gpu_info(self):
        """detect_cuda_gpu should return GPUInfo object."""
        info = detect_cuda_gpu()
        assert isinstance(info, GPUInfo)

    def test_returns_unavailable_when_no_backend(self, mocker):
        """detect_cuda_gpu should return unavailable when no backend works."""
        # Mock all backends to return unavailable
        mocker.patch.object(CuPyBackend, "get_gpu_info", return_value=GPUInfo(available=False))

        info = detect_cuda_gpu()
        assert info.available is False


class TestGetBackend:
    """Tests for get_backend function."""

    def test_returns_none_for_unknown_backend(self):
        """get_backend should return None for unknown backend name."""
        result = get_backend("nonexistent_backend")
        assert result is None

    def test_returns_cupy_backend_when_requested(self, mocker):
        """get_backend should return CuPyBackend when requested and available."""
        mocker.patch.object(CuPyBackend, "is_available", return_value=True)

        result = get_backend("cupy")
        assert result is not None
        assert isinstance(result, CuPyBackend)


class TestGetAvailableBackends:
    """Tests for get_available_backends function."""

    def test_returns_list(self):
        """get_available_backends should return a list."""
        result = get_available_backends()
        assert isinstance(result, list)

    def test_returns_cupy_when_available(self, mocker):
        """get_available_backends should include 'cupy' when CuPy is available."""
        mocker.patch.object(CuPyBackend, "is_available", return_value=True)

        result = get_available_backends()
        assert "cupy" in result

    def test_returns_empty_when_no_backends(self, mocker):
        """get_available_backends should return empty list when no backends available."""
        mocker.patch.object(CuPyBackend, "is_available", return_value=False)

        result = get_available_backends()
        assert result == []


class TestPrintGpuStatus:
    """Tests for print_gpu_status function."""

    def test_prints_gpu_detected_message_debug_mode(self, capsys):
        """print_gpu_status should print detailed GPU info in debug mode."""
        gpu_info = GPUInfo(
            available=True,
            name="Test GPU",
            cuda_version="13.0",
            driver_version="550.0",
            memory_total_mb=8192,
            memory_free_mb=7000,
            compute_capability="89",
            backend="cupy",
        )
        print_gpu_status(gpu_info, ProcessorType.GPU, debug=True)

        captured = capsys.readouterr()
        assert "Hardware Detection & Configuration" in captured.out
        assert "GPU Status:" in captured.out
        assert "Available" in captured.out
        assert "Test GPU" in captured.out
        assert "8192MB total" in captured.out
        assert "7000MB free" in captured.out
        assert "CUDA Version:" in captured.out
        assert "13.0" in captured.out
        assert "Driver Version:" in captured.out
        assert "550.0" in captured.out
        assert "Compute Capability:" in captured.out
        assert "89" in captured.out
        assert "Backend:" in captured.out
        assert "cupy" in captured.out
        assert "GPU acceleration enabled" in captured.out

    def test_prints_brief_gpu_message_normal_mode(self, capsys):
        """print_gpu_status should print brief message in normal mode."""
        gpu_info = GPUInfo(available=True, name="Test GPU", memory_total_mb=8192)
        print_gpu_status(gpu_info, ProcessorType.GPU, debug=False)

        captured = capsys.readouterr()
        assert "Using GPU acceleration" in captured.out
        assert "Test GPU" in captured.out
        # Should NOT contain detailed debug info
        assert "Hardware Detection & Configuration" not in captured.out

    def test_prints_cpu_fallback_message_debug_mode(self, capsys):
        """print_gpu_status should print detailed fallback message in debug mode."""
        gpu_info = GPUInfo(available=False)
        print_gpu_status(gpu_info, ProcessorType.CPU, debug=True)

        captured = capsys.readouterr()
        assert "Hardware Detection & Configuration" in captured.out
        assert "GPU Status:" in captured.out
        assert "Not available" in captured.out
        assert "CPU processing only" in captured.out
        assert "uv sync" in captured.out  # uv-first messaging

    def test_prints_brief_cpu_message_normal_mode(self, capsys):
        """print_gpu_status should print brief CPU message in normal mode."""
        gpu_info = GPUInfo(available=False)
        print_gpu_status(gpu_info, ProcessorType.CPU, debug=False)

        captured = capsys.readouterr()
        assert "Using CPU processing" in captured.out
        # Should NOT contain detailed debug info
        assert "Hardware Detection & Configuration" not in captured.out

    def test_prints_gpu_available_but_cpu_selected_debug_mode(self, capsys):
        """print_gpu_status should indicate when GPU available but CPU selected in debug mode."""
        gpu_info = GPUInfo(available=True, name="Test GPU", memory_total_mb=8192)
        print_gpu_status(gpu_info, ProcessorType.CPU, debug=True)

        captured = capsys.readouterr()
        assert "GPU available but not selected" in captured.out
        assert "CPU mode" in captured.out

    def test_prints_brief_cpu_when_gpu_available_normal_mode(self, capsys):
        """print_gpu_status should print brief CPU message when GPU available but not selected."""
        gpu_info = GPUInfo(available=True, name="Test GPU", memory_total_mb=8192)
        print_gpu_status(gpu_info, ProcessorType.CPU, debug=False)

        captured = capsys.readouterr()
        assert "Using CPU processing" in captured.out


# =============================================================================
# GPU Integration Tests (only run when GPU is available)
# =============================================================================


@pytest.mark.gpu
class TestCuPyBackendIntegration:
    """Integration tests for CuPyBackend that require actual GPU hardware."""

    def test_is_available_with_real_gpu(self):
        """CuPyBackend.is_available should return True on GPU systems."""
        backend = CuPyBackend()
        assert backend.is_available() is True

    def test_get_gpu_info_returns_valid_data(self):
        """get_gpu_info should return valid GPU information."""
        backend = CuPyBackend()
        info = backend.get_gpu_info()

        assert info.available is True
        assert info.name is not None
        assert info.cuda_version is not None
        assert info.memory_total_mb is not None
        assert info.memory_total_mb > 0
        assert info.compute_capability is not None
        assert len(info.compute_capability) == 2

    def test_to_gpu_and_to_cpu_roundtrip(self):
        """Data should survive GPU roundtrip unchanged."""
        import numpy as np

        backend = CuPyBackend()
        original = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

        gpu_array = backend.to_gpu(original)
        result = backend.to_cpu(gpu_array)

        np.testing.assert_array_equal(original, result)

    def test_configure_memory(self):
        """configure_memory should not raise errors."""
        backend = CuPyBackend()
        backend.configure_memory(0.5)  # Should not raise

    def test_free_memory(self):
        """free_memory should not raise errors."""
        backend = CuPyBackend()
        backend.free_memory()  # Should not raise
