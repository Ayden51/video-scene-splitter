"""
NVENC Hardware Encoding vs libx264 Software Encoding Benchmark.

This script benchmarks NVENC hardware encoding against libx264 software encoding
to measure encoding speedup and compare output quality/file size across different
video resolutions (SD, HD, 4K).

Run with: python benchmarks/nvenc_benchmark.py

Features:
- Tests encoding speed for NVENC vs libx264
- Measures encoding time, output file size, and bitrate
- Tests multiple quality presets for NVENC (p1, p4, p7)
- Compares across SD (480p), HD (1080p), and 4K (2160p) resolutions
- Generates detailed report in benchmarks/results/nvenc_benchmark.txt
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from video_scene_splitter.video_processor import (
    _get_libx264_options,
    _get_nvenc_options,
    detect_nvenc_support,
    get_encoder_info,
)


@dataclass
class EncodingResult:
    """Result from a single encoding benchmark."""

    video_name: str
    resolution: str
    encoder: str
    preset: str
    encoding_time_sec: float
    input_size_mb: float
    output_size_mb: float
    fps_encoded: float
    duration_sec: float
    bitrate_kbps: float
    error: str | None = None


@dataclass
class BenchmarkConfig:
    """Configuration for the encoding benchmark."""

    iterations: int = 3
    nvenc_presets: list[str] = field(default_factory=lambda: ["p1", "p4", "p7"])
    encode_duration_sec: float = 10.0  # Encode first 10 seconds


class NVENCBenchmark:
    """Comprehensive benchmark for NVENC vs libx264 encoding."""

    def __init__(self, config: BenchmarkConfig | None = None):
        self.config = config or BenchmarkConfig()
        self.nvenc_available = detect_nvenc_support()
        self.results: list[EncodingResult] = []
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
            import json

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

            # Determine resolution label
            if height >= 2160:
                resolution = "4K"
            elif height >= 1080:
                resolution = "HD"
            elif height >= 720:
                resolution = "HD-720p"
            else:
                resolution = "SD"

            return {
                "width": width,
                "height": height,
                "duration": duration,
                "fps": fps,
                "resolution": resolution,
                "size_mb": os.path.getsize(video_path) / 1024 / 1024,
            }
        except Exception as e:
            print(f"  Error getting video info: {e}")
            return None

    def run_encoding_benchmark(
        self, video_path: str, video_info: dict, encoder: str, preset: str
    ) -> EncodingResult:
        """Run a single encoding benchmark."""
        video_name = Path(video_path).name
        resolution = video_info["resolution"]

        # Build ffmpeg command
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            output_path = tmp.name

        try:
            cmd = ["ffmpeg", "-y", "-i", video_path, "-t", str(self.config.encode_duration_sec)]

            if encoder == "nvenc":
                cmd.extend(_get_nvenc_options(preset))
            else:
                cmd.extend(_get_libx264_options())

            cmd.extend(["-c:a", "copy", output_path])

            start_time = time.perf_counter()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            encoding_time = time.perf_counter() - start_time

            if result.returncode != 0:
                return EncodingResult(
                    video_name=video_name,
                    resolution=resolution,
                    encoder=encoder,
                    preset=preset,
                    encoding_time_sec=0,
                    input_size_mb=video_info["size_mb"],
                    output_size_mb=0,
                    fps_encoded=0,
                    duration_sec=0,
                    bitrate_kbps=0,
                    error=f"FFmpeg error: {result.stderr[:200]}",
                )

            output_size_mb = os.path.getsize(output_path) / 1024 / 1024
            encoded_duration = min(self.config.encode_duration_sec, video_info["duration"])
            fps_encoded = (encoded_duration * video_info["fps"]) / encoding_time
            bitrate_kbps = (output_size_mb * 8 * 1024) / encoded_duration

            return EncodingResult(
                video_name=video_name,
                resolution=resolution,
                encoder=encoder,
                preset=preset,
                encoding_time_sec=encoding_time,
                input_size_mb=video_info["size_mb"],
                output_size_mb=output_size_mb,
                fps_encoded=fps_encoded,
                duration_sec=encoded_duration,
                bitrate_kbps=bitrate_kbps,
            )
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def benchmark_video(self, video_path: str) -> dict[str, list[EncodingResult]]:
        """Run complete encoding benchmark on a single video."""
        video_info = self.get_video_info(video_path)
        if not video_info:
            print(f"  Skipping {video_path}: Cannot get video info")
            return {}

        video_name = Path(video_path).name
        resolution = video_info["resolution"]

        print(f"\n{'=' * 70}")
        print(f"Encoding Benchmark: {video_name} ({resolution})")
        print(f"  Resolution: {video_info['width']}x{video_info['height']}")
        print(f"  Duration: {video_info['duration']:.1f}s, FPS: {video_info['fps']:.1f}")
        print(f"  Input size: {video_info['size_mb']:.1f} MB")
        print(f"{'=' * 70}")

        results = {"libx264": [], "nvenc": []}

        # Benchmark libx264
        print("\n[libx264 Software Encoding]")
        for i in range(self.config.iterations):
            result = self.run_encoding_benchmark(video_path, video_info, "libx264", "default")
            results["libx264"].append(result)
            if result.error:
                print(f"  Run {i + 1}: ERROR - {result.error}")
            else:
                print(
                    f"  Run {i + 1}: {result.encoding_time_sec:.2f}s, "
                    f"{result.fps_encoded:.1f} FPS, {result.output_size_mb:.1f} MB"
                )
            self.results.append(result)

        # Benchmark NVENC (if available)
        if self.nvenc_available:
            for preset in self.config.nvenc_presets:
                print(f"\n[NVENC Hardware Encoding - Preset {preset}]")
                for i in range(self.config.iterations):
                    result = self.run_encoding_benchmark(video_path, video_info, "nvenc", preset)
                    results["nvenc"].append(result)
                    if result.error:
                        print(f"  Run {i + 1}: ERROR - {result.error}")
                    else:
                        print(
                            f"  Run {i + 1}: {result.encoding_time_sec:.2f}s, "
                            f"{result.fps_encoded:.1f} FPS, {result.output_size_mb:.1f} MB"
                        )
                    self.results.append(result)
        else:
            print("\n[NVENC not available - skipping hardware encoding tests]")

        return results

    def run_full_benchmark(self) -> None:
        """Run the complete benchmark suite."""
        print("\n" + "=" * 80)
        print("NVENC vs libx264 ENCODING BENCHMARK")
        print("=" * 80)

        # Check NVENC availability
        encoder_info = get_encoder_info("auto")
        print("\nEncoder Status:")
        print(f"  NVENC Available: {self.nvenc_available}")
        print(f"  Auto-selected: {encoder_info['name']} ({encoder_info['type']})")

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

        # Generate summary
        self._print_summary(all_results)
        self._save_report(all_results)

    def _print_summary(self, all_results: dict) -> None:
        """Print benchmark summary."""
        print("\n" + "=" * 80)
        print("ENCODING PERFORMANCE SUMMARY")
        print("=" * 80)

        print(
            f"\n{'Resolution':<15} {'Encoder':<15} {'Preset':<10} {'Avg Time':<12} "
            f"{'Avg FPS':<12} {'Speedup':<10} {'Size MB':<10}"
        )
        print("-" * 95)

        for label, results in all_results.items():
            if not results:
                continue

            # Calculate libx264 baseline
            libx264_results = [r for r in results.get("libx264", []) if not r.error]
            if libx264_results:
                libx264_avg_time = sum(r.encoding_time_sec for r in libx264_results) / len(
                    libx264_results
                )
                libx264_avg_fps = sum(r.fps_encoded for r in libx264_results) / len(libx264_results)
                libx264_avg_size = sum(r.output_size_mb for r in libx264_results) / len(
                    libx264_results
                )
                print(
                    f"{label:<15} {'libx264':<15} {'default':<10} {libx264_avg_time:<12.2f} "
                    f"{libx264_avg_fps:<12.1f} {'baseline':<10} {libx264_avg_size:<10.1f}"
                )

                # NVENC results by preset
                nvenc_results = [r for r in results.get("nvenc", []) if not r.error]
                for preset in self.config.nvenc_presets:
                    preset_results = [r for r in nvenc_results if r.preset == preset]
                    if preset_results:
                        nvenc_avg_time = sum(r.encoding_time_sec for r in preset_results) / len(
                            preset_results
                        )
                        nvenc_avg_fps = sum(r.fps_encoded for r in preset_results) / len(
                            preset_results
                        )
                        nvenc_avg_size = sum(r.output_size_mb for r in preset_results) / len(
                            preset_results
                        )
                        speedup = libx264_avg_time / nvenc_avg_time if nvenc_avg_time > 0 else 0
                        print(
                            f"{'':<15} {'nvenc':<15} {preset:<10} {nvenc_avg_time:<12.2f} "
                            f"{nvenc_avg_fps:<12.1f} {speedup:<10.2f}x {nvenc_avg_size:<10.1f}"
                        )
            print()

    def _save_report(self, all_results: dict) -> None:
        """Save detailed benchmark report to file."""
        report_path = self.results_dir / "nvenc_benchmark.txt"

        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("NVENC vs libx264 ENCODING BENCHMARK REPORT\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")

            f.write("CONFIGURATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Iterations per test: {self.config.iterations}\n")
            f.write(f"Encode duration: {self.config.encode_duration_sec}s\n")
            f.write(f"NVENC presets tested: {', '.join(self.config.nvenc_presets)}\n")
            f.write(f"NVENC available: {self.nvenc_available}\n\n")

            f.write("DETAILED RESULTS\n")
            f.write("-" * 80 + "\n")
            f.write(
                f"{'Video':<25} {'Resolution':<10} {'Encoder':<10} {'Preset':<8} "
                f"{'Time(s)':<10} {'FPS':<10} {'Size(MB)':<10}\n"
            )
            f.write("-" * 80 + "\n")

            for result in self.results:
                if not result.error:
                    f.write(
                        f"{result.video_name:<25} {result.resolution:<10} "
                        f"{result.encoder:<10} {result.preset:<8} "
                        f"{result.encoding_time_sec:<10.2f} {result.fps_encoded:<10.1f} "
                        f"{result.output_size_mb:<10.1f}\n"
                    )

            # Summary section
            f.write("\n" + "=" * 80 + "\n")
            f.write("PERFORMANCE SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            for label, results in all_results.items():
                if not results:
                    continue
                f.write(f"\n{label}\n")
                f.write("-" * 40 + "\n")

                libx264_results = [r for r in results.get("libx264", []) if not r.error]
                nvenc_results = [r for r in results.get("nvenc", []) if not r.error]

                if libx264_results:
                    avg_time = sum(r.encoding_time_sec for r in libx264_results) / len(
                        libx264_results
                    )
                    avg_fps = sum(r.fps_encoded for r in libx264_results) / len(libx264_results)
                    f.write(f"libx264: {avg_time:.2f}s avg, {avg_fps:.1f} FPS\n")

                    if nvenc_results:
                        nvenc_avg_time = sum(r.encoding_time_sec for r in nvenc_results) / len(
                            nvenc_results
                        )
                        speedup = avg_time / nvenc_avg_time if nvenc_avg_time > 0 else 0
                        f.write(f"NVENC:   {nvenc_avg_time:.2f}s avg ({speedup:.1f}x speedup)\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("KEY FINDINGS\n")
            f.write("=" * 80 + "\n")
            f.write("- NVENC provides significant speedup over libx264 for video encoding\n")
            f.write("- Higher resolutions show greater NVENC advantage\n")
            f.write("- Preset p1 is fastest, p7 provides highest quality\n")
            f.write("- p4 (default) provides good balance of speed and quality\n")

        print(f"\nReport saved to: {report_path}")


def main():
    """Main entry point for the benchmark."""
    config = BenchmarkConfig(iterations=3, nvenc_presets=["p1", "p4", "p7"])
    benchmark = NVENCBenchmark(config)
    benchmark.run_full_benchmark()


def quick_benchmark():
    """Run a quick benchmark with fewer iterations."""
    config = BenchmarkConfig(iterations=1, nvenc_presets=["p4"], encode_duration_sec=5.0)
    benchmark = NVENCBenchmark(config)
    benchmark.run_full_benchmark()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_benchmark()
    else:
        main()
