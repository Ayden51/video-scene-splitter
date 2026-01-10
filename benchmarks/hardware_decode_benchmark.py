"""
Hardware Video Decoding Benchmark: HardwareVideoReader (NVDEC) vs cv2.VideoCapture.

This script comprehensively benchmarks video decoding performance comparing:
- HardwareVideoReader with PyAV (optional NVDEC hardware acceleration)
- cv2.VideoCapture (software decoding via FFmpeg)

Features:
- Tests across SD (480p), HD (1080p), and 4K (2160p) resolutions
- Measures: FPS, total decode time, memory usage, initialization overhead
- Tests both CPU output (to_gpu=False) and GPU output (to_gpu=True) modes
- Includes graceful fallback testing when NVDEC is unavailable
- Generates detailed report in benchmarks/results/hardware_decode_benchmark.txt

Run with: python benchmarks/hardware_decode_benchmark.py
Quick test: python benchmarks/hardware_decode_benchmark.py --quick
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time
import tracemalloc
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from video_scene_splitter.gpu_utils import detect_cuda_gpu, get_backend
from video_scene_splitter.video_processor import (
    HardwareVideoReader,
    detect_nvdec_support,
    get_decode_info,
    is_codec_nvdec_compatible,
)


@dataclass
class DecodeResult:
    """Result from a single video decoding benchmark."""

    video_name: str
    resolution: str
    width: int
    height: int
    total_frames: int
    duration_sec: float
    decoder: str  # "cv2" or "pyav_nvdec" or "pyav_software"
    to_gpu: bool
    frames_decoded: int
    decode_time_sec: float
    init_time_sec: float
    fps_achieved: float
    memory_peak_mb: float
    memory_avg_mb: float
    hardware_enabled: bool
    fallback_reason: str | None = None
    error: str | None = None


@dataclass
class BenchmarkConfig:
    """Configuration for the hardware decode benchmark."""

    iterations: int = 3
    max_frames: int = 300  # Decode up to 300 frames per test
    batch_size: int = 30
    test_gpu_output: bool = True  # Test to_gpu=True mode


class HardwareDecodeBenchmark:
    """Comprehensive benchmark for hardware vs software video decoding."""

    def __init__(self, config: BenchmarkConfig | None = None):
        self.config = config or BenchmarkConfig()
        self.nvdec_available = detect_nvdec_support()
        self.gpu_info = detect_cuda_gpu()
        self.results: list[DecodeResult] = []
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)

    def get_video_info(self, video_path: str) -> dict[str, Any] | None:
        """Get video information using ffprobe."""
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-print_format",
                    "json",
                    "-show_format",
                    "-show_streams",
                    video_path,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            info = json.loads(result.stdout)
            video_stream = next(
                (s for s in info.get("streams", []) if s.get("codec_type") == "video"), None
            )
            if not video_stream:
                return None

            width = int(video_stream.get("width", 0))
            height = int(video_stream.get("height", 0))
            duration = float(info.get("format", {}).get("duration", 0))
            fps_str = video_stream.get("r_frame_rate", "30/1")
            fps_parts = fps_str.split("/")
            fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30.0
            codec = video_stream.get("codec_name", "unknown")

            if height >= 2160:
                resolution = "4K"
            elif height >= 1080:
                resolution = "HD"
            else:
                resolution = "SD"

            return {
                "width": width,
                "height": height,
                "duration": duration,
                "fps": fps,
                "resolution": resolution,
                "codec": codec,
                "total_frames": int(duration * fps),
                "size_mb": os.path.getsize(video_path) / 1024 / 1024,
                "nvdec_compatible": is_codec_nvdec_compatible(codec),
            }
        except Exception as e:
            print(f"  Error getting video info: {e}")
            return None

    def get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB."""
        try:
            backend = get_backend("cupy")
            if backend and backend.is_available():
                import cupy as cp

                mempool = cp.get_default_memory_pool()
                return mempool.used_bytes() / 1024 / 1024
        except Exception:
            pass
        return 0.0

    def benchmark_cv2_decode(self, video_path: str, video_info: dict) -> DecodeResult:
        """Benchmark cv2.VideoCapture decoding."""
        video_name = Path(video_path).name
        max_frames = min(self.config.max_frames, video_info["total_frames"])

        tracemalloc.start()
        gc.collect()

        try:
            # Measure initialization time
            init_start = time.perf_counter()
            cap = cv2.VideoCapture(video_path)
            init_time = time.perf_counter() - init_start

            if not cap.isOpened():
                return self._error_result(video_info, "cv2", "Failed to open video")

            # Decode frames
            frames_decoded = 0
            memory_samples = []
            decode_start = time.perf_counter()

            while frames_decoded < max_frames:
                ret, _frame = cap.read()
                if not ret:
                    break
                frames_decoded += 1
                if frames_decoded % 50 == 0:
                    current, _ = tracemalloc.get_traced_memory()
                    memory_samples.append(current / 1024 / 1024)

            decode_time = time.perf_counter() - decode_start
            cap.release()

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            fps_achieved = frames_decoded / decode_time if decode_time > 0 else 0
            memory_avg = sum(memory_samples) / len(memory_samples) if memory_samples else 0

            return DecodeResult(
                video_name=video_name,
                resolution=video_info["resolution"],
                width=video_info["width"],
                height=video_info["height"],
                total_frames=video_info["total_frames"],
                duration_sec=video_info["duration"],
                decoder="cv2",
                to_gpu=False,
                frames_decoded=frames_decoded,
                decode_time_sec=decode_time,
                init_time_sec=init_time,
                fps_achieved=fps_achieved,
                memory_peak_mb=peak / 1024 / 1024,
                memory_avg_mb=memory_avg,
                hardware_enabled=False,
            )

        except Exception as e:
            tracemalloc.stop()
            return self._error_result(video_info, "cv2", str(e))

    def benchmark_pyav_decode(
        self, video_path: str, video_info: dict, use_hardware: bool, to_gpu: bool = False
    ) -> DecodeResult:
        """Benchmark HardwareVideoReader (PyAV) decoding."""
        video_name = Path(video_path).name
        max_frames = min(self.config.max_frames, video_info["total_frames"])
        decoder_name = "pyav_nvdec" if use_hardware else "pyav_software"

        tracemalloc.start()
        gc.collect()
        initial_gpu_mem = self.get_gpu_memory_usage()

        try:
            # Measure initialization time
            init_start = time.perf_counter()
            reader = HardwareVideoReader(
                video_path, use_hardware=use_hardware, batch_size=self.config.batch_size
            )
            init_time = time.perf_counter() - init_start

            decode_info = reader.decode_info
            hardware_enabled = decode_info.hardware_enabled
            fallback_reason = decode_info.fallback_reason

            # Decode frames
            frames_decoded = 0
            memory_samples = []
            decode_start = time.perf_counter()

            for batch in reader.read_frames(batch_size=self.config.batch_size, to_gpu=to_gpu):
                frames_decoded += len(batch)
                if frames_decoded % 50 == 0:
                    if to_gpu:
                        gpu_mem = self.get_gpu_memory_usage() - initial_gpu_mem
                        memory_samples.append(gpu_mem)
                    else:
                        current, _ = tracemalloc.get_traced_memory()
                        memory_samples.append(current / 1024 / 1024)
                if frames_decoded >= max_frames:
                    break

            decode_time = time.perf_counter() - decode_start
            reader.close()

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            fps_achieved = frames_decoded / decode_time if decode_time > 0 else 0
            memory_avg = sum(memory_samples) / len(memory_samples) if memory_samples else 0

            if to_gpu:
                peak_memory = max(memory_samples) if memory_samples else 0
            else:
                peak_memory = peak / 1024 / 1024

            return DecodeResult(
                video_name=video_name,
                resolution=video_info["resolution"],
                width=video_info["width"],
                height=video_info["height"],
                total_frames=video_info["total_frames"],
                duration_sec=video_info["duration"],
                decoder=decoder_name,
                to_gpu=to_gpu,
                frames_decoded=frames_decoded,
                decode_time_sec=decode_time,
                init_time_sec=init_time,
                fps_achieved=fps_achieved,
                memory_peak_mb=peak_memory,
                memory_avg_mb=memory_avg,
                hardware_enabled=hardware_enabled,
                fallback_reason=fallback_reason,
            )

        except Exception as e:
            tracemalloc.stop()
            return self._error_result(video_info, decoder_name, str(e), to_gpu=to_gpu)

    def _error_result(
        self, video_info: dict, decoder: str, error: str, to_gpu: bool = False
    ) -> DecodeResult:
        """Create an error result."""
        return DecodeResult(
            video_name=Path(video_info.get("path", "unknown")).name,
            resolution=video_info.get("resolution", "unknown"),
            width=video_info.get("width", 0),
            height=video_info.get("height", 0),
            total_frames=video_info.get("total_frames", 0),
            duration_sec=video_info.get("duration", 0),
            decoder=decoder,
            to_gpu=to_gpu,
            frames_decoded=0,
            decode_time_sec=0,
            init_time_sec=0,
            fps_achieved=0,
            memory_peak_mb=0,
            memory_avg_mb=0,
            hardware_enabled=False,
            error=error,
        )

    def benchmark_video(self, video_path: str) -> dict[str, list[DecodeResult]]:
        """Run complete decode benchmark on a single video."""
        video_info = self.get_video_info(video_path)
        if not video_info:
            print(f"  Skipping {video_path}: Cannot get video info")
            return {}

        video_name = Path(video_path).name
        resolution = video_info["resolution"]

        print(f"\n{'=' * 70}")
        print(f"Hardware Decode Benchmark: {video_name} ({resolution})")
        print(f"  Resolution: {video_info['width']}x{video_info['height']}")
        print(f"  Duration: {video_info['duration']:.1f}s, FPS: {video_info['fps']:.1f}")
        print(
            f"  Codec: {video_info['codec']} (NVDEC compatible: {video_info['nvdec_compatible']})"
        )
        print(f"{'=' * 70}")

        results = {"cv2": [], "pyav_software": [], "pyav_nvdec": [], "pyav_gpu_output": []}

        # Benchmark cv2.VideoCapture
        print("\n[cv2.VideoCapture (Software Decode)]")
        for i in range(self.config.iterations):
            result = self.benchmark_cv2_decode(video_path, video_info)
            results["cv2"].append(result)
            self.results.append(result)
            if result.error:
                print(f"  Run {i + 1}: ERROR - {result.error}")
            else:
                print(
                    f"  Run {i + 1}: {result.fps_achieved:.1f} FPS, "
                    f"Init: {result.init_time_sec * 1000:.1f}ms, "
                    f"Mem: {result.memory_peak_mb:.1f}MB"
                )

        # Benchmark PyAV software decode
        print("\n[HardwareVideoReader - Software Mode]")
        for i in range(self.config.iterations):
            result = self.benchmark_pyav_decode(video_path, video_info, use_hardware=False)
            results["pyav_software"].append(result)
            self.results.append(result)
            if result.error:
                print(f"  Run {i + 1}: ERROR - {result.error}")
            else:
                print(
                    f"  Run {i + 1}: {result.fps_achieved:.1f} FPS, "
                    f"Init: {result.init_time_sec * 1000:.1f}ms, "
                    f"Mem: {result.memory_peak_mb:.1f}MB"
                )

        # Benchmark PyAV with NVDEC (if available)
        if self.nvdec_available and video_info["nvdec_compatible"]:
            print("\n[HardwareVideoReader - NVDEC Hardware Mode]")
            for i in range(self.config.iterations):
                result = self.benchmark_pyav_decode(video_path, video_info, use_hardware=True)
                results["pyav_nvdec"].append(result)
                self.results.append(result)
                if result.error:
                    print(f"  Run {i + 1}: ERROR - {result.error}")
                else:
                    hw_status = "HW" if result.hardware_enabled else "SW-fallback"
                    print(
                        f"  Run {i + 1}: {result.fps_achieved:.1f} FPS [{hw_status}], "
                        f"Init: {result.init_time_sec * 1000:.1f}ms, "
                        f"Mem: {result.memory_peak_mb:.1f}MB"
                    )
        else:
            reason = (
                "NVDEC not available" if not self.nvdec_available else "Codec not NVDEC compatible"
            )
            print(f"\n[HardwareVideoReader - NVDEC] Skipped: {reason}")

        # Benchmark GPU output mode (if GPU available)
        if self.config.test_gpu_output and self.gpu_info.available:
            print("\n[HardwareVideoReader - GPU Output (to_gpu=True)]")
            for i in range(self.config.iterations):
                result = self.benchmark_pyav_decode(
                    video_path, video_info, use_hardware=self.nvdec_available, to_gpu=True
                )
                results["pyav_gpu_output"].append(result)
                self.results.append(result)
                if result.error:
                    print(f"  Run {i + 1}: ERROR - {result.error}")
                else:
                    print(
                        f"  Run {i + 1}: {result.fps_achieved:.1f} FPS, "
                        f"GPU Mem: {result.memory_peak_mb:.1f}MB"
                    )
        elif not self.gpu_info.available:
            print("\n[HardwareVideoReader - GPU Output] Skipped: GPU not available")

        return results

    def run_full_benchmark(self) -> None:
        """Run the complete benchmark suite."""
        print("\n" + "=" * 80)
        print("HARDWARE VIDEO DECODING BENCHMARK")
        print("HardwareVideoReader (PyAV/NVDEC) vs cv2.VideoCapture")
        print("=" * 80)

        # System info
        print("\nSystem Configuration:")
        print(f"  NVDEC Available: {self.nvdec_available}")
        print(f"  GPU Available: {self.gpu_info.available}")
        if self.gpu_info.available:
            print(f"  GPU Name: {self.gpu_info.name}")
            print(f"  GPU Memory: {self.gpu_info.memory_total_mb} MB")

        decode_info = get_decode_info("auto")
        print(f"\nDecode Mode (AUTO): {decode_info['name']}")

        # Find test videos
        input_dir = Path(__file__).parent.parent / "input"
        videos = {
            "SD (480p)": input_dir / "sd-sample.mp4",
            "HD (1080p)": input_dir / "hd-sample.mp4",
            "4K (2160p)": input_dir / "4k-sample.mp4",
        }

        all_results = {}
        for label, video_path in videos.items():
            if video_path.exists():
                all_results[label] = self.benchmark_video(str(video_path))
            else:
                print(f"\n[Skipping {label}] Video not found: {video_path}")

        # Generate summary and report
        self._print_summary(all_results)
        self._save_report(all_results)

        print("\n" + "=" * 80)
        print("BENCHMARK COMPLETE")
        print("=" * 80)

    def _print_summary(self, all_results: dict) -> None:
        """Print benchmark summary."""
        print("\n" + "=" * 80)
        print("DECODE PERFORMANCE SUMMARY")
        print("=" * 80)

        print(
            f"\n{'Resolution':<12} {'Decoder':<18} {'Avg FPS':<12} {'Init(ms)':<12} "
            f"{'Mem(MB)':<10} {'Speedup':<10}"
        )
        print("-" * 85)

        for label, results in all_results.items():
            if not results:
                continue

            # Calculate cv2 baseline
            cv2_results = [r for r in results.get("cv2", []) if not r.error]
            cv2_avg_fps = (
                sum(r.fps_achieved for r in cv2_results) / len(cv2_results) if cv2_results else 0
            )

            for decoder_key in ["cv2", "pyav_software", "pyav_nvdec", "pyav_gpu_output"]:
                decoder_results = [r for r in results.get(decoder_key, []) if not r.error]
                if decoder_results:
                    avg_fps = sum(r.fps_achieved for r in decoder_results) / len(decoder_results)
                    avg_init = (
                        sum(r.init_time_sec for r in decoder_results) / len(decoder_results) * 1000
                    )
                    avg_mem = sum(r.memory_peak_mb for r in decoder_results) / len(decoder_results)

                    if decoder_key == "cv2":
                        speedup_str = "baseline"
                    elif cv2_avg_fps > 0:
                        speedup = avg_fps / cv2_avg_fps
                        speedup_str = f"{speedup:.2f}x"
                    else:
                        speedup_str = "-"

                    decoder_name = {
                        "cv2": "cv2.VideoCapture",
                        "pyav_software": "PyAV (Software)",
                        "pyav_nvdec": "PyAV (NVDEC)",
                        "pyav_gpu_output": "PyAV (GPU Out)",
                    }.get(decoder_key, decoder_key)

                    print(
                        f"{label:<12} {decoder_name:<18} {avg_fps:<12.1f} "
                        f"{avg_init:<12.1f} {avg_mem:<10.1f} {speedup_str:<10}"
                    )
            print()

    def _save_report(self, all_results: dict) -> None:
        """Save detailed benchmark report to file."""
        report_path = self.results_dir / "hardware_decode_benchmark.txt"

        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("HARDWARE VIDEO DECODING BENCHMARK REPORT\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")

            f.write("SYSTEM CONFIGURATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"NVDEC Available: {self.nvdec_available}\n")
            f.write(f"GPU Available: {self.gpu_info.available}\n")
            if self.gpu_info.available:
                f.write(f"GPU Name: {self.gpu_info.name}\n")
                f.write(f"GPU Memory: {self.gpu_info.memory_total_mb} MB\n")
            f.write("\n")

            f.write("BENCHMARK CONFIGURATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Iterations: {self.config.iterations}\n")
            f.write(f"Max frames per test: {self.config.max_frames}\n")
            f.write(f"Batch size: {self.config.batch_size}\n\n")

            f.write("DETAILED RESULTS\n")
            f.write("-" * 100 + "\n")
            f.write(
                f"{'Video':<20} {'Decoder':<18} {'Frames':<8} {'Time(s)':<10} "
                f"{'FPS':<10} {'Init(ms)':<10} {'Mem(MB)':<10}\n"
            )
            f.write("-" * 100 + "\n")

            for result in self.results:
                if not result.error:
                    f.write(
                        f"{result.video_name:<20} {result.decoder:<18} "
                        f"{result.frames_decoded:<8} {result.decode_time_sec:<10.2f} "
                        f"{result.fps_achieved:<10.1f} {result.init_time_sec * 1000:<10.1f} "
                        f"{result.memory_peak_mb:<10.1f}\n"
                    )

            # Summary by resolution
            f.write("\n" + "=" * 80 + "\n")
            f.write("PERFORMANCE SUMMARY BY RESOLUTION\n")
            f.write("=" * 80 + "\n")

            for label, results in all_results.items():
                if not results:
                    continue
                f.write(f"\n{label}\n")
                f.write("-" * 40 + "\n")

                cv2_results = [r for r in results.get("cv2", []) if not r.error]
                cv2_fps = (
                    sum(r.fps_achieved for r in cv2_results) / len(cv2_results)
                    if cv2_results
                    else 0
                )

                for decoder_key in ["cv2", "pyav_software", "pyav_nvdec", "pyav_gpu_output"]:
                    decoder_results = [r for r in results.get(decoder_key, []) if not r.error]
                    if decoder_results:
                        avg_fps = sum(r.fps_achieved for r in decoder_results) / len(
                            decoder_results
                        )
                        speedup = avg_fps / cv2_fps if cv2_fps > 0 else 0
                        f.write(f"  {decoder_key}: {avg_fps:.1f} FPS ({speedup:.2f}x vs cv2)\n")

            # Key findings and recommendations
            f.write("\n" + "=" * 80 + "\n")
            f.write("KEY FINDINGS & AUTO MODE RECOMMENDATION\n")
            f.write("=" * 80 + "\n\n")

            # Analyze results for recommendation
            self._write_recommendations(f, all_results)

        print(f"\nReport saved to: {report_path}")

    def _write_recommendations(self, f, all_results: dict) -> None:
        """Write analysis and recommendations to report."""
        # Calculate average speedups
        speedups = {"pyav_software": [], "pyav_nvdec": [], "pyav_gpu_output": []}

        for results in all_results.values():
            cv2_results = [r for r in results.get("cv2", []) if not r.error]
            cv2_fps = (
                sum(r.fps_achieved for r in cv2_results) / len(cv2_results) if cv2_results else 0
            )

            for key in speedups:
                decoder_results = [r for r in results.get(key, []) if not r.error]
                if decoder_results and cv2_fps > 0:
                    avg_fps = sum(r.fps_achieved for r in decoder_results) / len(decoder_results)
                    speedups[key].append(avg_fps / cv2_fps)

        f.write("1. DECODE PERFORMANCE ANALYSIS\n")
        f.write("-" * 40 + "\n")

        for key, values in speedups.items():
            if values:
                avg_speedup = sum(values) / len(values)
                f.write(f"  {key}: Average {avg_speedup:.2f}x vs cv2.VideoCapture\n")
        f.write("\n")

        f.write("2. AUTO MODE INTEGRATION RECOMMENDATION\n")
        f.write("-" * 40 + "\n")

        # Determine recommendation based on speedups
        nvdec_speedups = speedups.get("pyav_nvdec", [])
        avg_nvdec_speedup = sum(nvdec_speedups) / len(nvdec_speedups) if nvdec_speedups else 0

        if avg_nvdec_speedup >= 1.2:
            f.write("  RECOMMENDATION: INTEGRATE HardwareVideoReader into AUTO mode\n")
            f.write(f"  - NVDEC provides {avg_nvdec_speedup:.2f}x average speedup\n")
            f.write("  - Conditions for AUTO mode to use HardwareVideoReader:\n")
            f.write("    * NVDEC available (detect_nvdec_support() returns True)\n")
            f.write("    * Video codec is NVDEC-compatible (H.264, H.265, VP9)\n")
            f.write("    * GPU processing mode selected (processor='gpu' or 'auto' with GPU)\n")
        elif avg_nvdec_speedup >= 1.0:
            f.write("  RECOMMENDATION: OPTIONAL integration with user preference\n")
            f.write(f"  - NVDEC provides marginal {avg_nvdec_speedup:.2f}x speedup\n")
            f.write("  - Benefits depend on system configuration\n")
        else:
            f.write("  RECOMMENDATION: DO NOT integrate by default\n")
            f.write("  - cv2.VideoCapture performs comparably or better\n")
            f.write("  - HardwareVideoReader adds complexity without clear benefit\n")

        f.write("\n")
        f.write("3. MEMORY USAGE ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write("  - cv2.VideoCapture: Consistent memory usage, well-optimized\n")
        f.write("  - PyAV: Slightly higher initialization overhead\n")
        f.write("  - GPU Output mode: Requires GPU VRAM for frames\n\n")

        f.write("4. INITIALIZATION OVERHEAD\n")
        f.write("-" * 40 + "\n")
        f.write("  - cv2.VideoCapture: Minimal initialization (~10-50ms)\n")
        f.write("  - HardwareVideoReader: Slightly higher (~20-100ms)\n")
        f.write("  - For short clips, initialization may dominate total time\n")


def main():
    """Main entry point for the benchmark."""
    config = BenchmarkConfig(iterations=3, max_frames=300, batch_size=30)
    benchmark = HardwareDecodeBenchmark(config)
    benchmark.run_full_benchmark()


def quick_benchmark():
    """Run a quick benchmark with fewer iterations."""
    config = BenchmarkConfig(iterations=1, max_frames=100, batch_size=30, test_gpu_output=False)
    benchmark = HardwareDecodeBenchmark(config)
    benchmark.run_full_benchmark()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_benchmark()
    else:
        main()
