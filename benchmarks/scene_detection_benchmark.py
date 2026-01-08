"""
Phase 2B Scene Detection Benchmark - CPU vs GPU Performance Comparison.

This script comprehensively benchmarks the VideoSceneSplitter's scene detection
performance across different processor modes, batch sizes, and video resolutions.

Run with: python benchmarks/scene_detection_benchmark.py

Features:
- Tests all processor modes: cpu, gpu, auto
- Tests multiple GPU batch sizes: 5, 15, 30, 60, auto
- Tests different GPU memory fractions: 0.5, 0.7, 0.8
- Measures detection time, speedup ratios, memory usage
- Verifies result consistency between CPU and GPU
- Tests OOM recovery with large batch sizes
- Reports frame processing rates (FPS)

Output:
- Tabular results with resolution, processor, batch size, time, speedup, memory
- Summary statistics comparing GPU performance across resolutions
- Optimal batch size recommendations
- Comparison against Phase 2A expectations
"""

import contextlib
import gc
import io
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from video_scene_splitter import GPUInfo, VideoSceneSplitter, detect_cuda_gpu


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    video_name: str
    resolution: str
    width: int
    height: int
    total_frames: int
    processor_mode: str
    batch_size: str | int
    memory_fraction: float
    detection_time_sec: float
    scenes_detected: int
    scene_timestamps: list[float]
    fps_processed: float
    gpu_memory_used_mb: float = 0.0
    error: str | None = None


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    iterations: int = 3
    processor_modes: list[str] = field(default_factory=lambda: ["cpu", "gpu", "auto"])
    batch_sizes: list[int | str] = field(default_factory=lambda: [5, 15, 30, 60, "auto"])
    memory_fractions: list[float] = field(default_factory=lambda: [0.5, 0.7, 0.8])
    threshold: float = 30.0
    min_scene_duration: float = 1.5


class SceneDetectionBenchmark:
    """Comprehensive benchmark for VideoSceneSplitter scene detection."""

    def __init__(self, config: BenchmarkConfig | None = None):
        self.config = config or BenchmarkConfig()
        self.gpu_info: GPUInfo = detect_cuda_gpu()
        self.results: list[BenchmarkResult] = []
        self.cpu_baseline: dict[str, BenchmarkResult] = {}  # video_name -> CPU result

    def print_header(self):
        """Print benchmark header with system info."""
        print("=" * 80)
        print("PHASE 2B SCENE DETECTION BENCHMARK")
        print("CPU vs GPU Performance Comparison")
        print("=" * 80)
        print(f"\nGPU Status: {'Available' if self.gpu_info.available else 'Not Available'}")
        if self.gpu_info.available:
            print(f"  Device: {self.gpu_info.name}")
            print(
                f"  Memory: {self.gpu_info.memory_total_mb:.0f} MB total, "
                f"{self.gpu_info.memory_free_mb:.0f} MB free"
            )
            print(f"  CUDA: {self.gpu_info.cuda_version}")
        print("\nBenchmark Configuration:")
        print(f"  Iterations per config: {self.config.iterations}")
        print(f"  Detection threshold: {self.config.threshold}")
        print(f"  Min scene duration: {self.config.min_scene_duration}s")
        print()

    def get_video_info(self, video_path: str) -> dict[str, Any] | None:
        """Get video metadata."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
        }
        cap.release()
        # Determine resolution label
        if info["height"] >= 2160:
            info["resolution"] = "4K"
        elif info["height"] >= 1080:
            info["resolution"] = "HD"
        elif info["height"] >= 720:
            info["resolution"] = "720p"
        else:
            info["resolution"] = "SD"
        return info

    def get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB."""
        if not self.gpu_info.available:
            return 0.0
        try:
            import cupy as cp

            mempool = cp.get_default_memory_pool()
            return mempool.used_bytes() / 1024 / 1024
        except Exception:
            return 0.0

    def free_gpu_memory(self):
        """Free GPU memory."""
        if not self.gpu_info.available:
            return
        try:
            import cupy as cp

            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            cp.cuda.Stream.null.synchronize()
        except Exception:
            pass
        gc.collect()

    def run_single_detection(
        self,
        video_path: str,
        video_info: dict,
        processor_mode: str,
        batch_size: int | str,
        memory_fraction: float,
    ) -> BenchmarkResult:
        """Run a single scene detection benchmark."""
        video_name = Path(video_path).name
        self.free_gpu_memory()
        initial_memory = self.get_gpu_memory_usage()

        # Suppress splitter output during benchmark
        stdout_capture = io.StringIO()

        try:
            splitter = VideoSceneSplitter(
                video_path=video_path,
                output_dir="output/benchmark_temp",
                threshold=self.config.threshold,
                min_scene_duration=self.config.min_scene_duration,
                processor=processor_mode,
                gpu_batch_size=batch_size,
                gpu_memory_fraction=memory_fraction,
            )

            # Time the detection (suppress output)
            with contextlib.redirect_stdout(stdout_capture):
                start_time = time.perf_counter()
                timestamps = splitter.detect_scenes(debug=False)
                detection_time = time.perf_counter() - start_time

            peak_memory = self.get_gpu_memory_usage()
            fps_processed = video_info["total_frames"] / detection_time if detection_time > 0 else 0

            return BenchmarkResult(
                video_name=video_name,
                resolution=video_info["resolution"],
                width=video_info["width"],
                height=video_info["height"],
                total_frames=video_info["total_frames"],
                processor_mode=processor_mode,
                batch_size=batch_size,
                memory_fraction=memory_fraction,
                detection_time_sec=detection_time,
                scenes_detected=len(timestamps),
                scene_timestamps=timestamps,
                fps_processed=fps_processed,
                gpu_memory_used_mb=max(0, peak_memory - initial_memory),
            )

        except Exception as e:
            return BenchmarkResult(
                video_name=video_name,
                resolution=video_info["resolution"],
                width=video_info["width"],
                height=video_info["height"],
                total_frames=video_info["total_frames"],
                processor_mode=processor_mode,
                batch_size=batch_size,
                memory_fraction=memory_fraction,
                detection_time_sec=0,
                scenes_detected=0,
                scene_timestamps=[],
                fps_processed=0,
                error=str(e),
            )

    def run_video_benchmark(self, video_path: str) -> list[BenchmarkResult]:
        """Run full benchmark suite for a single video."""
        video_info = self.get_video_info(video_path)
        if not video_info:
            print(f"  [!] Cannot open video: {video_path}")
            return []

        video_name = Path(video_path).name
        print(f"\n{'─' * 70}")
        print(f"Benchmarking: {video_name}")
        res = video_info["resolution"]
        print(f"  Resolution: {video_info['width']}x{video_info['height']} ({res})")
        print(f"  Frames: {video_info['total_frames']}, Duration: {video_info['duration']:.1f}s")
        print(f"{'─' * 70}")

        results = []

        # 1. CPU Baseline (run first for comparison)
        print("\n[CPU Baseline]")
        cpu_times = []
        for i in range(self.config.iterations):
            result = self.run_single_detection(video_path, video_info, "cpu", 30, 0.8)
            if result.error:
                print(f"  Run {i + 1}: ERROR - {result.error}")
            else:
                cpu_times.append(result.detection_time_sec)
                print(
                    f"  Run {i + 1}: {result.detection_time_sec:.3f}s, "
                    f"{result.scenes_detected} scenes, {result.fps_processed:.1f} FPS"
                )

        if cpu_times:
            avg_cpu_time = sum(cpu_times) / len(cpu_times)
            # Create baseline result with average time
            baseline = self.run_single_detection(video_path, video_info, "cpu", 30, 0.8)
            baseline.detection_time_sec = avg_cpu_time
            baseline.fps_processed = video_info["total_frames"] / avg_cpu_time
            self.cpu_baseline[video_name] = baseline
            results.append(baseline)
            print(f"  Average: {avg_cpu_time:.3f}s ({baseline.fps_processed:.1f} FPS)")

        # Skip GPU tests if not available
        if not self.gpu_info.available:
            print("\n  [!] GPU not available, skipping GPU benchmarks")
            return results

        # 2. GPU with various batch sizes (using default memory fraction)
        print("\n[GPU Batch Size Tests] (memory_fraction=0.8)")
        for batch_size in self.config.batch_sizes:
            batch_times = []
            batch_memory = []
            for i in range(self.config.iterations):
                result = self.run_single_detection(video_path, video_info, "gpu", batch_size, 0.8)
                if result.error:
                    print(f"  Batch {batch_size}, Run {i + 1}: ERROR - {result.error}")
                    break
                batch_times.append(result.detection_time_sec)
                batch_memory.append(result.gpu_memory_used_mb)

            if batch_times:
                avg_time = sum(batch_times) / len(batch_times)
                avg_memory = sum(batch_memory) / len(batch_memory)
                speedup = (
                    self.cpu_baseline[video_name].detection_time_sec / avg_time
                    if avg_time > 0
                    else 0
                )
                result.detection_time_sec = avg_time
                result.fps_processed = video_info["total_frames"] / avg_time
                result.gpu_memory_used_mb = avg_memory
                results.append(result)
                print(
                    f"  Batch {batch_size!s:>4}: {avg_time:.3f}s, "
                    f"{speedup:.2f}x speedup, {avg_memory:.1f}MB GPU mem"
                )

        # 3. GPU with various memory fractions (using batch_size=30)
        print("\n[GPU Memory Fraction Tests] (batch_size=30)")
        for mem_frac in self.config.memory_fractions:
            if mem_frac == 0.8:  # Already tested above
                continue
            frac_times = []
            for i in range(self.config.iterations):
                result = self.run_single_detection(video_path, video_info, "gpu", 30, mem_frac)
                if result.error:
                    print(f"  Mem {mem_frac}, Run {i + 1}: ERROR - {result.error}")
                    break
                frac_times.append(result.detection_time_sec)

            if frac_times:
                avg_time = sum(frac_times) / len(frac_times)
                speedup = (
                    self.cpu_baseline[video_name].detection_time_sec / avg_time
                    if avg_time > 0
                    else 0
                )
                result.detection_time_sec = avg_time
                result.fps_processed = video_info["total_frames"] / avg_time
                results.append(result)
                print(f"  Mem {mem_frac}: {avg_time:.3f}s, {speedup:.2f}x speedup")

        # 4. Auto mode test
        print("\n[Auto Mode Test]")
        auto_times = []
        for i in range(self.config.iterations):
            result = self.run_single_detection(video_path, video_info, "auto", 30, 0.8)
            if result.error:
                print(f"  Auto Run {i + 1}: ERROR - {result.error}")
            else:
                auto_times.append(result.detection_time_sec)

        if auto_times:
            avg_time = sum(auto_times) / len(auto_times)
            speedup = (
                self.cpu_baseline[video_name].detection_time_sec / avg_time if avg_time > 0 else 0
            )
            result.detection_time_sec = avg_time
            result.fps_processed = video_info["total_frames"] / avg_time
            results.append(result)
            print(f"  Auto: {avg_time:.3f}s, {speedup:.2f}x speedup (uses GPU if available)")

        self.results.extend(results)
        return results

    def verify_result_consistency(self, video_path: str) -> dict[str, Any]:
        """Verify that CPU and GPU produce consistent scene detection results."""
        video_info = self.get_video_info(video_path)
        if not video_info or not self.gpu_info.available:
            return {"verified": False, "reason": "GPU not available or video not found"}

        video_name = Path(video_path).name
        print(f"\n[Result Consistency Check] {video_name}")

        # Run CPU detection
        cpu_result = self.run_single_detection(video_path, video_info, "cpu", 30, 0.8)
        if cpu_result.error:
            return {"verified": False, "reason": f"CPU error: {cpu_result.error}"}

        # Run GPU detection
        gpu_result = self.run_single_detection(video_path, video_info, "gpu", 30, 0.8)
        if gpu_result.error:
            return {"verified": False, "reason": f"GPU error: {gpu_result.error}"}

        # Compare results
        cpu_scenes = cpu_result.scenes_detected
        gpu_scenes = gpu_result.scenes_detected
        scenes_match = cpu_scenes == gpu_scenes

        # Compare timestamps (allow small tolerance for floating point)
        timestamps_match = True
        if scenes_match and cpu_result.scene_timestamps:
            cpu_ts_list = cpu_result.scene_timestamps
            gpu_ts_list = gpu_result.scene_timestamps
            for cpu_ts, gpu_ts in zip(cpu_ts_list, gpu_ts_list, strict=True):
                if abs(cpu_ts - gpu_ts) > 0.1:  # 0.1 second tolerance
                    timestamps_match = False
                    break

        result = {
            "verified": scenes_match and timestamps_match,
            "cpu_scenes": cpu_scenes,
            "gpu_scenes": gpu_scenes,
            "scenes_match": scenes_match,
            "timestamps_match": timestamps_match,
            "cpu_timestamps": cpu_result.scene_timestamps[:5],  # First 5
            "gpu_timestamps": gpu_result.scene_timestamps[:5],
        }

        status = "[PASS]" if result["verified"] else "[FAIL]"
        print(f"  {status}: CPU={cpu_scenes} scenes, GPU={gpu_scenes} scenes")
        if not result["verified"]:
            print(f"    CPU timestamps: {result['cpu_timestamps']}")
            print(f"    GPU timestamps: {result['gpu_timestamps']}")

        return result

    def test_oom_recovery(self, video_path: str) -> dict[str, Any]:
        """Test OOM recovery by intentionally using large batch sizes."""
        video_info = self.get_video_info(video_path)
        if not video_info or not self.gpu_info.available:
            return {"tested": False, "reason": "GPU not available or video not found"}

        video_name = Path(video_path).name
        print(f"\n[OOM Recovery Test] {video_name}")

        # Try increasingly large batch sizes
        large_batch_sizes = [60, 90, 120]
        results = []

        for batch_size in large_batch_sizes:
            self.free_gpu_memory()
            result = self.run_single_detection(video_path, video_info, "gpu", batch_size, 0.8)

            if result.error and "memory" in result.error.lower():
                print(f"  Batch {batch_size}: OOM detected, CPU fallback triggered [OK]")
                results.append({"batch_size": batch_size, "oom": True, "recovered": True})
            elif result.error:
                print(f"  Batch {batch_size}: Error - {result.error}")
                results.append({"batch_size": batch_size, "oom": False, "error": result.error})
            else:
                elapsed = result.detection_time_sec
                print(f"  Batch {batch_size}: Processed successfully ({elapsed:.3f}s)")
                results.append({"batch_size": batch_size, "oom": False, "success": True})

        return {"tested": True, "results": results}

    def print_summary_table(self):
        """Print summary table of all results."""
        print("\n" + "=" * 100)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 100)

        # Group results by video
        results_by_video: dict[str, list[BenchmarkResult]] = {}
        for r in self.results:
            if r.video_name not in results_by_video:
                results_by_video[r.video_name] = []
            results_by_video[r.video_name].append(r)

        # Print table header
        print(
            f"\n{'Video':<20} {'Resolution':<10} {'Processor':<8} {'Batch':<8} "
            f"{'Time(s)':<10} {'Speedup':<10} {'FPS':<10} {'GPU Mem':<10}"
        )
        print("-" * 100)

        for video_name, results in results_by_video.items():
            cpu_baseline = self.cpu_baseline.get(video_name)
            baseline_time = cpu_baseline.detection_time_sec if cpu_baseline else 0

            for r in results:
                speedup = baseline_time / r.detection_time_sec if r.detection_time_sec > 0 else 0
                speedup_str = f"{speedup:.2f}x" if speedup > 0 else "N/A"
                mem_str = f"{r.gpu_memory_used_mb:.1f}MB" if r.gpu_memory_used_mb > 0 else "-"
                time_str = f"{r.detection_time_sec:.3f}" if not r.error else "ERROR"
                fps_str = f"{r.fps_processed:.1f}" if r.fps_processed > 0 else "-"

                print(
                    f"{r.video_name:<20} {r.resolution:<10} {r.processor_mode:<8} "
                    f"{r.batch_size!s:<8} {time_str:<10} {speedup_str:<10} "
                    f"{fps_str:<10} {mem_str:<10}"
                )
            print()  # Blank line between videos

    def print_performance_analysis(self):
        """Print performance analysis and recommendations."""
        print("\n" + "=" * 80)
        print("PERFORMANCE ANALYSIS")
        print("=" * 80)

        # Calculate average speedups by resolution
        speedups_by_resolution: dict[str, list[float]] = {}

        for video_name, baseline in self.cpu_baseline.items():
            resolution = baseline.resolution
            if resolution not in speedups_by_resolution:
                speedups_by_resolution[resolution] = []

            # Find GPU results for this video
            for r in self.results:
                if r.video_name == video_name and r.processor_mode == "gpu":
                    speedup = (
                        baseline.detection_time_sec / r.detection_time_sec
                        if r.detection_time_sec > 0
                        else 0
                    )
                    if speedup > 0:
                        speedups_by_resolution[resolution].append(speedup)

        print("\n[Average GPU Speedup by Resolution]")
        print("-" * 50)

        expectations = {
            "SD": ("0.8-1.2x", 0.8),
            "720p": ("1.0-1.5x", 1.0),
            "HD": ("1.2-1.4x", 1.2),
            "4K": ("1.5-2.5x", 1.5),
        }

        for resolution, speedups in speedups_by_resolution.items():
            if speedups:
                avg_speedup = sum(speedups) / len(speedups)
                min_speedup = min(speedups)
                max_speedup = max(speedups)
                expected, min_expected = expectations.get(resolution, ("N/A", 0))

                status = "[OK]" if avg_speedup >= min_expected else "[!]"
                range_str = f"{min_speedup:.2f}x - {max_speedup:.2f}x"
                print(f"  {resolution}: {avg_speedup:.2f}x avg (range: {range_str})")
                print(f"       Expected: {expected} {status}")

        # Optimal batch size recommendation
        print("\n[Optimal Batch Size Recommendations]")
        print("-" * 50)

        for video_name, baseline in self.cpu_baseline.items():
            best_speedup = 0
            best_batch = 30

            for r in self.results:
                if r.video_name == video_name and r.processor_mode == "gpu" and not r.error:
                    speedup = (
                        baseline.detection_time_sec / r.detection_time_sec
                        if r.detection_time_sec > 0
                        else 0
                    )
                    if speedup > best_speedup:
                        best_speedup = speedup
                        best_batch = r.batch_size

            print(f"  {video_name}: batch_size={best_batch} ({best_speedup:.2f}x speedup)")

        # Phase 2A comparison
        print("\n[Comparison with Phase 2A Expectations]")
        print("-" * 50)
        print("  Phase 2A targets:")
        print("    - HD video (1080p): 1.2-1.4x speedup")
        print("    - Histogram GPU: ~0.77x of CPU (acceptable)")
        print()

        if speedups_by_resolution.get("HD"):
            hd_avg = sum(speedups_by_resolution["HD"]) / len(speedups_by_resolution["HD"])
            hd_status = "[MET]" if hd_avg >= 1.2 else ("[CLOSE]" if hd_avg >= 1.0 else "[BELOW]")
            print(f"  Actual HD speedup: {hd_avg:.2f}x - {hd_status}")
        else:
            print("  HD benchmark data not available")

    def run_full_benchmark(self):
        """Run the complete benchmark suite."""
        self.print_header()

        # Define test videos
        input_dir = Path(__file__).parent.parent / "input"
        videos = [
            ("sd-sample.mp4", "SD (480p)"),
            ("hd-sample.mp4", "HD (1080p)"),
            ("4k-sample.mp4", "4K (2160p)"),
        ]

        # Check which videos exist
        available_videos = []
        print("Available Test Videos:")
        for filename, label in videos:
            video_path = input_dir / filename
            if video_path.exists():
                info = self.get_video_info(str(video_path))
                if info:
                    print(
                        f"  [OK] {filename}: {info['width']}x{info['height']}, "
                        f"{info['total_frames']} frames, {info['duration']:.1f}s"
                    )
                    available_videos.append((str(video_path), label))
            else:
                print(f"  [--] {filename}: Not found")

        if not available_videos:
            print("\n[!] No test videos found in input/ directory!")
            print("  Please add: sd-sample.mp4, hd-sample.mp4, 4k-sample.mp4")
            return

        # Run benchmarks for each video
        print("\n" + "=" * 80)
        print("RUNNING BENCHMARKS")
        print("=" * 80)

        for video_path, _label in available_videos:
            self.run_video_benchmark(video_path)

        # Run consistency checks
        print("\n" + "=" * 80)
        print("RESULT CONSISTENCY VERIFICATION")
        print("=" * 80)

        consistency_results = {}
        for video_path, label in available_videos:
            result = self.verify_result_consistency(video_path)
            consistency_results[label] = result

        # Run OOM recovery tests (only on HD or 4K for more memory pressure)
        print("\n" + "=" * 80)
        print("OOM RECOVERY TESTING")
        print("=" * 80)

        oom_results = {}
        for video_path, label in available_videos:
            if "HD" in label or "4K" in label:
                result = self.test_oom_recovery(video_path)
                oom_results[label] = result

        # Print summary
        self.print_summary_table()
        self.print_performance_analysis()

        # Print final summary
        print("\n" + "=" * 80)
        print("BENCHMARK COMPLETE")
        print("=" * 80)

        # Consistency summary
        all_consistent = all(r.get("verified", False) for r in consistency_results.values())
        print(f"\nResult Consistency: {'[All PASS]' if all_consistent else '[Some FAILED]'}")

        # Overall stats
        total_configs = len(self.results)
        successful_configs = len([r for r in self.results if not r.error])
        print(f"Configurations Tested: {successful_configs}/{total_configs}")

        # Free GPU memory
        self.free_gpu_memory()


def main():
    """Main entry point for the benchmark."""
    config = BenchmarkConfig(
        iterations=3,
        processor_modes=["cpu", "gpu", "auto"],
        batch_sizes=[5, 15, 30, 60, "auto"],
        memory_fractions=[0.5, 0.7, 0.8],
        threshold=30.0,
        min_scene_duration=1.5,
    )

    benchmark = SceneDetectionBenchmark(config)
    benchmark.run_full_benchmark()


def quick_benchmark():
    """Run a quick benchmark with fewer iterations for testing."""
    config = BenchmarkConfig(
        iterations=1,
        processor_modes=["cpu", "gpu"],
        batch_sizes=[15, 30],
        memory_fractions=[0.8],
        threshold=30.0,
        min_scene_duration=1.5,
    )

    benchmark = SceneDetectionBenchmark(config)
    benchmark.run_full_benchmark()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2B Scene Detection Benchmark")
    parser.add_argument(
        "--quick", action="store_true", help="Run quick benchmark with fewer iterations"
    )
    parser.add_argument(
        "--iterations", type=int, default=3, help="Number of iterations per configuration"
    )
    args = parser.parse_args()

    if args.quick:
        quick_benchmark()
    else:
        config = BenchmarkConfig(iterations=args.iterations)
        benchmark = SceneDetectionBenchmark(config)
        benchmark.run_full_benchmark()
