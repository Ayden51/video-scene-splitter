"""
GPU vs CPU benchmark for scene detection algorithms.

This script benchmarks the GPU-accelerated detection functions against
their CPU counterparts to measure speedup and verify result accuracy.

Run with: python benchmarks/gpu_benchmark.py
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from video_scene_splitter.detection import (
    compute_histogram_distance,
    compute_pixel_difference,
)
from video_scene_splitter.detection_gpu import (
    compute_histogram_distance_batch_gpu,
    compute_histogram_distance_gpu,
    compute_pixel_difference_batch_gpu,
    compute_pixel_difference_gpu,
    free_gpu_memory,
)
from video_scene_splitter.gpu_utils import detect_cuda_gpu


def generate_test_frames(num_frames: int, height: int = 480, width: int = 640) -> list:
    """Generate random test frames for benchmarking."""
    np.random.seed(42)
    frames = []
    for i in range(num_frames):
        # Create frames with some structure (not pure noise)
        base_color = (i * 10) % 256
        frame = np.full((height, width, 3), base_color, dtype=np.uint8)
        # Add some random variation
        noise = np.random.randint(-20, 20, (height, width, 3))
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frames.append(frame)
    return frames


def benchmark_single_frame_pair(frame1, frame2, iterations: int = 100):
    """Benchmark single frame pair processing."""
    print("\n=== Single Frame Pair Benchmark ===")

    # CPU pixel difference
    start = time.perf_counter()
    for _ in range(iterations):
        cpu_mean, cpu_ratio = compute_pixel_difference(frame1, frame2)
    cpu_pixel_time = (time.perf_counter() - start) / iterations * 1000

    # GPU pixel difference
    start = time.perf_counter()
    for _ in range(iterations):
        gpu_mean, gpu_ratio = compute_pixel_difference_gpu(frame1, frame2)
    gpu_pixel_time = (time.perf_counter() - start) / iterations * 1000

    print(f"Pixel Difference (single pair, {iterations} iterations):")
    print(f"  CPU: {cpu_pixel_time:.3f} ms/pair")
    print(f"  GPU: {gpu_pixel_time:.3f} ms/pair")
    print(f"  Speedup: {cpu_pixel_time / gpu_pixel_time:.2f}x")
    mean_match = abs(cpu_mean - gpu_mean) < 0.01
    ratio_match = abs(cpu_ratio - gpu_ratio) < 0.001
    print(f"  Result match: mean_diff={mean_match}, ratio={ratio_match}")

    # CPU histogram distance
    start = time.perf_counter()
    for _ in range(iterations):
        cpu_dist = compute_histogram_distance(frame1, frame2)
    cpu_hist_time = (time.perf_counter() - start) / iterations * 1000

    # GPU histogram distance
    start = time.perf_counter()
    for _ in range(iterations):
        gpu_dist = compute_histogram_distance_gpu(frame1, frame2)
    gpu_hist_time = (time.perf_counter() - start) / iterations * 1000

    print(f"\nHistogram Distance (single pair, {iterations} iterations):")
    print(f"  CPU: {cpu_hist_time:.3f} ms/pair")
    print(f"  GPU: {gpu_hist_time:.3f} ms/pair")
    print(f"  Speedup: {cpu_hist_time / gpu_hist_time:.2f}x")
    print(f"  Result match: distance_diff={abs(cpu_dist - gpu_dist):.4f}")

    return {
        "pixel_cpu": cpu_pixel_time,
        "pixel_gpu": gpu_pixel_time,
        "hist_cpu": cpu_hist_time,
        "hist_gpu": gpu_hist_time,
    }


def benchmark_batch_processing(frames: list, batch_sizes: list | None = None):
    """Benchmark batch processing with various sizes."""
    if batch_sizes is None:
        batch_sizes = [5, 15, 30, 60]
    print("\n=== Batch Processing Benchmark ===")

    results = []
    for batch_size in batch_sizes:
        if batch_size > len(frames):
            continue

        batch = frames[:batch_size]
        iterations = max(1, 50 // batch_size)  # Fewer iterations for large batches

        # CPU sequential pixel difference
        start = time.perf_counter()
        for _ in range(iterations):
            for i in range(len(batch) - 1):
                compute_pixel_difference(batch[i], batch[i + 1])
        cpu_pixel_time = (time.perf_counter() - start) / iterations * 1000

        # GPU batch pixel difference
        start = time.perf_counter()
        for _ in range(iterations):
            compute_pixel_difference_batch_gpu(batch)
        gpu_pixel_time = (time.perf_counter() - start) / iterations * 1000

        # CPU sequential histogram distance
        start = time.perf_counter()
        for _ in range(iterations):
            for i in range(len(batch) - 1):
                compute_histogram_distance(batch[i], batch[i + 1])
        cpu_hist_time = (time.perf_counter() - start) / iterations * 1000

        # GPU batch histogram distance
        start = time.perf_counter()
        for _ in range(iterations):
            compute_histogram_distance_batch_gpu(batch)
        gpu_hist_time = (time.perf_counter() - start) / iterations * 1000

        print(f"\nBatch size: {batch_size} frames ({batch_size - 1} pairs)")
        pixel_speedup = cpu_pixel_time / gpu_pixel_time
        hist_speedup = cpu_hist_time / gpu_hist_time
        print(
            f"  Pixel Diff - CPU: {cpu_pixel_time:.2f}ms, GPU: {gpu_pixel_time:.2f}ms, "
            f"Speedup: {pixel_speedup:.2f}x"
        )
        print(
            f"  Histogram  - CPU: {cpu_hist_time:.2f}ms, GPU: {gpu_hist_time:.2f}ms, "
            f"Speedup: {hist_speedup:.2f}x"
        )

        results.append(
            {
                "batch_size": batch_size,
                "pixel_speedup": cpu_pixel_time / gpu_pixel_time,
                "hist_speedup": cpu_hist_time / gpu_hist_time,
            }
        )

    return results


def benchmark_with_video(video_path: str, max_frames: int = 100, label: str = ""):
    """Benchmark using real video frames."""
    video_name = Path(video_path).name
    print(f"\n{'=' * 60}")
    print(f"Video Benchmark: {video_name} {label}")
    print(f"{'=' * 60}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Error: Cannot open video {video_path}")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"  Resolution: {width}x{height}")
    print(f"  Total frames: {total_frames}, FPS: {fps:.1f}")
    print(f"  Frame size: {width * height * 3 / 1024 / 1024:.2f} MB")

    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    print(f"  Loaded {len(frames)} frames for benchmark")

    if len(frames) < 2:
        print("  Not enough frames for benchmark")
        return None

    # Determine safe batch sizes based on frame size
    frame_mb = width * height * 3 / 1024 / 1024
    if frame_mb > 20:  # 4K frames ~24MB each
        batch_sizes = [5, 10, 15]
    elif frame_mb > 5:  # HD frames ~6MB each
        batch_sizes = [5, 15, 30]
    else:  # SD frames ~1.2MB each
        batch_sizes = [5, 15, 30, 60]

    return benchmark_batch_processing(frames, batch_sizes)


def benchmark_memory_usage(frames: list, batch_size: int = 30):
    """Measure GPU memory usage during processing."""
    try:
        import cupy as cp

        # Get initial memory
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        initial_used = mempool.used_bytes() / 1024 / 1024

        # Process batch and measure peak
        batch = frames[: min(batch_size, len(frames))]

        # Run pixel difference
        compute_pixel_difference_batch_gpu(batch)
        after_pixel = mempool.used_bytes() / 1024 / 1024

        mempool.free_all_blocks()

        # Run histogram distance
        compute_histogram_distance_batch_gpu(batch)
        after_hist = mempool.used_bytes() / 1024 / 1024

        mempool.free_all_blocks()
        final_used = mempool.used_bytes() / 1024 / 1024

        return {
            "initial_mb": initial_used,
            "after_pixel_mb": after_pixel,
            "after_hist_mb": after_hist,
            "final_mb": final_used,
            "batch_size": len(batch),
        }
    except Exception as e:
        print(f"  Memory measurement error: {e}")
        return None


def run_comprehensive_video_benchmark():
    """Run comprehensive benchmark on all available video samples."""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE REAL VIDEO BENCHMARK")
    print("=" * 70)

    input_dir = Path(__file__).parent.parent / "input"
    videos = {
        "SD (480p)": input_dir / "sd-sample.mp4",
        "HD (1080p)": input_dir / "hd-sample.mp4",
        "4K (2160p)": input_dir / "4k-sample.mp4",
    }

    results = {}
    memory_results = {}

    for label, video_path in videos.items():
        if video_path.exists():
            # Run performance benchmark
            result = benchmark_with_video(str(video_path), max_frames=65, label=label)
            if result:
                results[label] = result

            # Run memory benchmark
            print(f"\n  Memory Usage Test for {label}:")
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            for _ in range(35):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()

            if frames:
                mem_result = benchmark_memory_usage(frames, batch_size=30)
                if mem_result:
                    memory_results[label] = mem_result
                    print(f"    Batch of {mem_result['batch_size']} frames:")
                    print(f"    After pixel diff: {mem_result['after_pixel_mb']:.1f} MB GPU memory")
                    print(f"    After histogram:  {mem_result['after_hist_mb']:.1f} MB GPU memory")
        else:
            print(f"\n  Video not found: {video_path}")

    # Print summary comparison
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Resolution':<15} {'Batch':<8} {'Pixel Speedup':<15} {'Hist Speedup':<15}")
    print("-" * 53)

    for label, batch_results in results.items():
        for r in batch_results:
            px_spd = f"{r['pixel_speedup']:.2f}x"
            hist_spd = f"{r['hist_speedup']:.2f}x"
            print(f"{label:<15} {r['batch_size']:<8} {px_spd:<15} {hist_spd}")

    # Calculate average speedups per resolution
    print("\n" + "=" * 70)
    print("AVERAGE SPEEDUPS BY RESOLUTION")
    print("=" * 70)
    for label, batch_results in results.items():
        avg_pixel = sum(r["pixel_speedup"] for r in batch_results) / len(batch_results)
        avg_hist = sum(r["hist_speedup"] for r in batch_results) / len(batch_results)
        overall = (avg_pixel + avg_hist) / 2
        print(f"{label}: Pixel={avg_pixel:.2f}x, Histogram={avg_hist:.2f}x, Overall={overall:.2f}x")

    # Memory summary
    if memory_results:
        print("\n" + "=" * 70)
        print("MEMORY USAGE SUMMARY (30-frame batch)")
        print("=" * 70)
        for label, mem in memory_results.items():
            print(f"{label}: Peak={max(mem['after_pixel_mb'], mem['after_hist_mb']):.1f} MB")

    return results, memory_results


def main():
    print("=" * 60)
    print("GPU vs CPU Benchmark for Video Scene Detection")
    print("=" * 60)

    # Check GPU availability
    gpu_info = detect_cuda_gpu()
    print(f"\nGPU Status: {'Available' if gpu_info.available else 'Not Available'}")
    if gpu_info.available:
        print(f"  Device: {gpu_info.name}")
        print(f"  Memory: {gpu_info.memory_total_mb:.0f} MB total")
        print(f"  Free Memory: {gpu_info.memory_free_mb:.0f} MB")
        print(f"  CUDA: {gpu_info.cuda_version}")
    else:
        print("  GPU benchmarks will be skipped!")
        return

    # Generate test frames for synthetic benchmark
    print("\nGenerating synthetic test frames...")
    frames = generate_test_frames(65, height=480, width=640)

    # Run synthetic benchmarks
    single_results = benchmark_single_frame_pair(frames[0], frames[1])
    batch_results = benchmark_batch_processing(frames)

    # Summary of synthetic benchmarks
    print("\n" + "=" * 60)
    print("SYNTHETIC FRAME BENCHMARK SUMMARY")
    print("=" * 60)
    pixel_speedup = single_results["pixel_cpu"] / single_results["pixel_gpu"]
    hist_speedup = single_results["hist_cpu"] / single_results["hist_gpu"]
    print(f"Single pair pixel diff speedup: {pixel_speedup:.2f}x")
    print(f"Single pair histogram speedup: {hist_speedup:.2f}x")
    if batch_results:
        avg_pixel = sum(r["pixel_speedup"] for r in batch_results) / len(batch_results)
        avg_hist = sum(r["hist_speedup"] for r in batch_results) / len(batch_results)
        print(f"Average batch pixel diff speedup: {avg_pixel:.2f}x")
        print(f"Average batch histogram speedup: {avg_hist:.2f}x")

    # Run comprehensive video benchmarks if videos exist
    input_dir = Path(__file__).parent.parent / "input"
    if any((input_dir / f).exists() for f in ["sd-sample.mp4", "hd-sample.mp4", "4k-sample.mp4"]):
        video_results, _memory_results = run_comprehensive_video_benchmark()

        # Print expectations comparison
        print("\n" + "=" * 70)
        print("COMPARISON WITH PHASE 2A EXPECTATIONS")
        print("=" * 70)
        expectations = {
            "SD (480p)": "1-2x speedup (CPU competitive)",
            "HD (1080p)": "2-4x speedup",
            "4K (2160p)": "4-8x speedup",
        }

        for label, batch_results in video_results.items():
            avg_pixel = sum(r["pixel_speedup"] for r in batch_results) / len(batch_results)
            avg_hist = sum(r["hist_speedup"] for r in batch_results) / len(batch_results)
            overall = (avg_pixel + avg_hist) / 2
            expected = expectations.get(label, "N/A")
            status = "✓" if overall >= 1.0 else "✗"
            print(f"{label}:")
            print(f"  Expected: {expected}")
            print(f"  Actual:   {overall:.2f}x overall ({status})")

    # Free GPU memory
    free_gpu_memory()
    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
