"""
Main VideoSceneSplitter class that orchestrates scene detection and video splitting.

This module provides the high-level interface for the video scene splitter,
coordinating between detection algorithms, video processing, and utilities.
"""

import time
from pathlib import Path

import cv2

from .detection import (
    compute_histogram_distance,
    compute_pixel_difference,
    is_hard_cut,
)
from .gpu_utils import (
    AutoModeConfig,
    GPUConfig,
    GPUInfo,
    ProcessorType,
    detect_cuda_gpu,
    estimate_optimal_batch_size,
    get_resolution_batch_size_cap,
    print_gpu_status,
    select_operation_processor,
    select_processor,
    should_use_async_io,
)
from .utils import save_debug_frames, save_metrics_to_csv, save_timestamps_to_file
from .video_processor import read_frames_async, split_video_at_timestamps


class VideoSceneSplitter:
    """
    Intelligent video scene splitter that detects and splits videos at hard cuts.

    This class analyzes video content frame-by-frame using multiple computer vision
    algorithms to identify hard cuts—moments where the entire visual content changes
    abruptly. It then splits the original video into separate files for each scene.

    Attributes:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where output files will be saved.
        threshold (float): Detection sensitivity threshold (0-100 scale).
        min_scene_duration (float): Minimum scene length in seconds.
        scene_timestamps (list): List of detected scene start times in seconds.

    Example:
        >>> splitter = VideoSceneSplitter(
        ...     video_path="input/video.mp4",
        ...     threshold=20.0,
        ...     min_scene_duration=0.5
        ... )
        >>> splitter.detect_scenes(debug=True)
        >>> splitter.split_video()
    """

    def __init__(
        self,
        video_path,
        output_dir="output",
        threshold=30.0,
        min_scene_duration=1.5,
        processor="auto",
        gpu_batch_size=30,
        gpu_memory_fraction=0.8,
    ):
        """
        Initialize the video scene splitter for hard cut detection.

        Args:
            video_path (str): Path to the input video file (relative or absolute).
                Supported formats: MP4, AVI, MOV, MKV, and other OpenCV-compatible formats.
            output_dir (str, optional): Directory to save split videos and debug files.
                Defaults to "output". Created automatically if it doesn't exist.
            threshold (float, optional): Hard cut detection sensitivity threshold.
                Defaults to 30.0. Typical range: 15-40.
                - Lower values (15-20): More sensitive, may detect subtle changes
                - Medium values (20-30): Balanced, good for most videos
                - Higher values (30-40): Conservative, only obvious hard cuts
                - Values above 40: Very conservative, may miss some cuts
            min_scene_duration (float, optional): Minimum scene duration in seconds.
                Defaults to 1.5. Prevents detection of very short scenes.
                Useful for filtering out brief flashes or transitions.
            processor (str, optional): Processing mode for scene detection.
                Defaults to "auto". Options:
                - "auto": Automatically detect and use GPU if available, fallback to CPU
                - "cpu": Force CPU-only processing (useful for debugging or compatibility)
                - "gpu": Force GPU processing (raises error if GPU unavailable)
            gpu_batch_size (int or str, optional): Number of frames to process in each
                GPU batch. Defaults to 30. Options:
                - Integer (5-120): Fixed batch size
                - "auto": Automatically determine based on GPU memory and frame size
                Larger batches are more efficient but use more GPU memory.
                Recommended values by GPU VRAM:
                - 4 GB: 15 frames
                - 8 GB: 30 frames
                - 12+ GB: 60 frames
            gpu_memory_fraction (float, optional): Maximum fraction of GPU memory to use.
                Defaults to 0.8 (80%). Range: 0.1-1.0.
                Lower values leave more memory for other applications.

        Raises:
            ValueError: If video_path doesn't exist or threshold is negative.
            RuntimeError: If processor="gpu" but no GPU is available.

        Note:
            The threshold operates on a 0-100 scale where higher values indicate
            greater difference between frames. Run with debug=True to see statistics
            and get a suggested threshold for your specific video.
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.threshold = threshold
        self.min_scene_duration = min_scene_duration
        self.scene_timestamps = []

        # GPU configuration
        self._gpu_batch_size = gpu_batch_size
        self._gpu_memory_fraction = gpu_memory_fraction
        self._gpu_config = GPUConfig(
            batch_size=gpu_batch_size if isinstance(gpu_batch_size, int) else 30,
            memory_fraction=gpu_memory_fraction,
        )

        # GPU detection and processor selection
        self._gpu_info: GPUInfo = detect_cuda_gpu()
        self._processor_request = ProcessorType.from_string(processor)
        self._active_processor: ProcessorType = select_processor(
            self._processor_request, self._gpu_info
        )

    def detect_scenes(self, debug=False):
        """
        Detect hard cuts (scene changes) in the video using multi-metric analysis.

        This method analyzes the video frame-by-frame, comparing consecutive frames
        using histogram distance, pixel difference, and changed pixel ratio. A hard
        cut is detected when all three metrics exceed their thresholds simultaneously,
        indicating an abrupt change in visual content.

        When GPU acceleration is enabled (processor="gpu" or "auto" with GPU available),
        frames are processed in batches for improved performance. The batch size can
        be configured via the gpu_batch_size parameter.

        The detection process:
        1. Reads video frames sequentially (or in batches for GPU mode)
        2. Computes three metrics for each frame pair:
           - HSV histogram distance (color content change)
           - Mean pixel difference (brightness/intensity change)
           - Changed pixel ratio (spatial change coverage)
        3. Identifies hard cuts when all conditions are met:
           - Histogram distance > threshold
           - Pixel difference > threshold
           - Changed pixel ratio > 20%
           - Time since last cut >= min_scene_duration
        4. Records timestamps of detected scene boundaries

        Args:
            debug (bool, optional): Enable debug mode for detailed analysis.
                Defaults to False. When True:
                - Displays detailed GPU hardware information and acceleration status
                - Saves before/after frames at each detected cut as JPEG images
                - Generates CSV file with frame-by-frame metrics
                - Prints statistical analysis (min, max, mean, std dev)
                - Suggests optimal threshold based on video statistics

        Returns:
            list: List of scene start timestamps in seconds (float).
                Always includes 0.0 as the first scene start.
                Example: [0.0, 5.23, 12.45, 18.67]

        Raises:
            ValueError: If video file cannot be opened or first frame cannot be read.

        Example:
            >>> splitter = VideoSceneSplitter("video.mp4", threshold=25.0)
            >>> timestamps = splitter.detect_scenes(debug=True)
            Analyzing video: video.mp4
            Video: 30.00 FPS, 900 frames, 30.00s
            ✓ Scene 2 at 5.23s (frame 157)
            ✓ Scene 3 at 12.45s (frame 374)
            >>> print(timestamps)
            [0.0, 5.23, 12.45]

        Note:
            Progress is displayed every 100 frames. For long videos, this provides
            feedback that processing is ongoing. The final output includes scene
            durations and statistics when debug mode is enabled.
        """
        print(f"Analyzing video: {self.video_path}")
        Path(self.output_dir).mkdir(exist_ok=True)

        # Print GPU/processor status (detailed in debug mode, brief otherwise)
        print_gpu_status(self._gpu_info, self._active_processor, debug=debug)

        # Dispatch to appropriate implementation based on processor mode
        if self._processor_request == ProcessorType.AUTO and self._gpu_info.available:
            # AUTO mode with GPU available: use hybrid processing
            return self._detect_scenes_hybrid(debug=debug)
        elif self._active_processor == ProcessorType.GPU:
            # Pure GPU mode
            return self._detect_scenes_gpu(debug=debug)
        else:
            # CPU mode (or AUTO without GPU)
            return self._detect_scenes_cpu(debug=debug)

    def _detect_scenes_cpu(self, debug=False):
        """
        CPU implementation of scene detection (original algorithm).

        This method processes frames one at a time using CPU-based detection
        algorithms from the detection module.
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        min_frame_gap = int(self.min_scene_duration * fps)

        print(f"Video: {fps:.2f} FPS, {total_frames} frames, {duration:.2f}s")
        print(f"Threshold: {self.threshold}")
        print(f"Min scene duration: {self.min_scene_duration}s ({min_frame_gap} frames)")

        # Read first frame
        ret, prev_frame = cap.read()
        if not ret:
            raise ValueError("Cannot read first frame")

        frame_num = 1
        last_scene_frame = 0
        self.scene_timestamps = [0.0]

        all_metrics = []

        print("\n" + "=" * 70)
        print("Detecting hard cuts...")
        print("=" * 70)

        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break

            # Compute multiple metrics using detection algorithms
            hist_dist = compute_histogram_distance(prev_frame, curr_frame)
            pixel_diff, changed_ratio = compute_pixel_difference(prev_frame, curr_frame)

            metrics = {
                "frame": frame_num,
                "timestamp": frame_num / fps,
                "hist_dist": hist_dist,
                "pixel_diff": pixel_diff,
                "changed_ratio": changed_ratio * 100,  # Convert to percentage
            }
            all_metrics.append(metrics)

            # Check for hard cut using detection algorithm
            frames_since_last = frame_num - last_scene_frame

            if is_hard_cut(
                hist_dist,
                pixel_diff,
                changed_ratio,
                self.threshold,
                frames_since_last,
                min_frame_gap,
            ):
                timestamp = frame_num / fps
                self.scene_timestamps.append(timestamp)

                print(
                    f"✓ Scene {len(self.scene_timestamps)} at {timestamp:.2f}s (frame {frame_num})"
                )
                print(
                    f"  Hist: {hist_dist:.1f}, Pixel: {pixel_diff:.1f}, "
                    f"Changed: {changed_ratio * 100:.1f}%"
                )

                last_scene_frame = frame_num

                # Save debug frames if requested
                if debug:
                    save_debug_frames(
                        self.output_dir,
                        len(self.scene_timestamps),
                        frame_num - 1,
                        frame_num,
                        prev_frame,
                        curr_frame,
                    )

            prev_frame = curr_frame.copy()
            frame_num += 1

            # Progress indicator
            if frame_num % 100 == 0:
                progress = (frame_num / total_frames) * 100
                print(f"Progress: {progress:.1f}%", end="\r")

        cap.release()

        self._print_detection_summary(duration, debug, all_metrics)
        return self.scene_timestamps

    def _detect_scenes_hybrid(self, debug=False):
        """
        Hybrid CPU/GPU scene detection with per-operation processor selection and async I/O.

        This method implements performance-optimized processing based on actual benchmark
        results (2026-01-09):

        Scene Detection:
        - SD (480p): GPU pixel diff (1.32x overall speedup)
        - HD (1080p): GPU pixel diff (1.39x overall speedup, best benefit)
        - 4K (2160p): CPU pixel diff (GPU 0.88x, slower due to transfer overhead)
        - Histogram: Always CPU (1.1-1.7x faster than GPU)

        Frame Reading:
        - GPU mode: Async I/O (1.01-1.54x speedup when GPU time >= 10ms)
        - CPU mode: Sync I/O (no async benefit, avoid overhead)
        """
        # Import GPU functions (lazy import to avoid errors when GPU unavailable)
        from .detection_gpu import (
            compute_pixel_difference_batch_gpu,
            free_gpu_memory,
        )

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        min_frame_gap = int(self.min_scene_duration * fps)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Classify resolution for logging
        if frame_height >= 2160:
            resolution_class = "4K"
        elif frame_height >= 720:
            resolution_class = "HD"
        else:
            resolution_class = "SD"

        print(f"Video: {fps:.2f} FPS, {total_frames} frames, {duration:.2f}s")
        print(f"Resolution: {frame_width}x{frame_height} ({resolution_class})")
        print(f"Threshold: {self.threshold}")
        print(f"Min scene duration: {self.min_scene_duration}s ({min_frame_gap} frames)")

        # Determine per-operation processor selection with logging
        auto_config = AutoModeConfig()
        print("\n🔧 AUTO mode decisions (benchmark-based):")

        hist_processor = select_operation_processor(
            "histogram", frame_height, frame_width, self._gpu_info, auto_config, verbose=True
        )
        pixel_processor = select_operation_processor(
            "pixel_diff", frame_height, frame_width, self._gpu_info, auto_config, verbose=True
        )

        # Determine async I/O usage
        use_async = should_use_async_io(pixel_processor, self._gpu_info, auto_config, verbose=True)

        print(
            f"\n🔧 Hybrid mode: Histogram={hist_processor.value.upper()}, "
            f"PixelDiff={pixel_processor.value.upper()}"
        )

        # Determine batch size based on pixel processor
        if pixel_processor == ProcessorType.GPU:
            batch_size = self._determine_batch_size(frame_height, frame_width, debug)
            # Apply resolution-based cap
            batch_size = min(batch_size, get_resolution_batch_size_cap(frame_height, auto_config))
        else:
            # CPU batch size - smaller batches for memory efficiency
            batch_size = 30

        print(f"Batch size: {batch_size}")
        io_mode = "async (I/O overlap)" if use_async else "sync (sequential)"
        print(f"Frame reading: {io_mode}")

        # Read first frame
        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("Cannot read first frame")

        frame_num = 1
        last_scene_frame = 0
        self.scene_timestamps = [0.0]
        all_metrics = []

        print("\n" + "=" * 70)
        mode_desc = "hybrid CPU/GPU" if pixel_processor == ProcessorType.GPU else "CPU"
        io_desc = "async I/O" if use_async else "sync I/O"
        print(f"Detecting hard cuts ({mode_desc} with {io_desc})...")
        print("=" * 70)

        # Frame buffer for batch processing - start with first frame
        frame_buffer = [first_frame]
        frame_indices = [0]

        # Choose frame reading method based on async decision
        if use_async:
            frame_reader = read_frames_async(cap, batch_size=batch_size)
        else:
            frame_reader = self._read_frames_sync(cap, batch_size)

        # Process frames using selected I/O method
        for batch in frame_reader:
            # Add batch frames to buffer
            for frame in batch:
                frame_buffer.append(frame)
                frame_indices.append(frame_num)
                frame_num += 1

            # Process batch when buffer is full (batch_size + 1 for overlap)
            while len(frame_buffer) >= batch_size + 1:
                last_scene_frame = self._process_hybrid_batch(
                    frame_buffer,
                    frame_indices,
                    fps,
                    min_frame_gap,
                    last_scene_frame,
                    all_metrics,
                    debug,
                    total_frames,
                    hist_processor,
                    pixel_processor,
                    compute_pixel_difference_batch_gpu,
                    free_gpu_memory,
                )

                # Keep last frame for next batch overlap
                frame_buffer = [frame_buffer[-1]]
                frame_indices = [frame_indices[-1]]

        # Process remaining frames in buffer
        if len(frame_buffer) > 1:
            self._process_hybrid_batch(
                frame_buffer,
                frame_indices,
                fps,
                min_frame_gap,
                last_scene_frame,
                all_metrics,
                debug,
                total_frames,
                hist_processor,
                pixel_processor,
                compute_pixel_difference_batch_gpu,
                free_gpu_memory,
            )

        cap.release()

        self._print_detection_summary(duration, debug, all_metrics)
        return self.scene_timestamps

    def _detect_scenes_gpu(self, debug=False):
        """
        GPU-accelerated scene detection with batch processing.

        This method processes frames in batches on the GPU for improved performance.
        It uses the GPU frame pool pattern to minimize CPU-GPU data transfers:
        1. Upload a batch of frames to GPU once
        2. Process both pixel difference and histogram distance on the same batch
        3. Transfer only the results back to CPU
        4. Free GPU memory before processing next batch

        Uses cv2.VideoCapture for frame reading (CPU decode) with async I/O to overlap
        disk reads with GPU computation. Frames are read as NumPy arrays and
        transferred to GPU for batch processing.
        """
        # Import GPU functions (lazy import to avoid errors when GPU unavailable)
        from .detection_gpu import (
            _stack_frames_to_gpu,
            compute_histogram_distance_batch_gpu,
            compute_pixel_difference_batch_gpu,
            free_gpu_memory,
        )

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        min_frame_gap = int(self.min_scene_duration * fps)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video: {fps:.2f} FPS, {total_frames} frames, {duration:.2f}s")
        print(f"Threshold: {self.threshold}")
        print(f"Min scene duration: {self.min_scene_duration}s ({min_frame_gap} frames)")

        # Determine batch size
        batch_size = self._determine_batch_size(frame_height, frame_width, debug)

        print(f"GPU batch size: {batch_size}")
        print("Frame decoding: cv2 (CPU) → GPU upload (CuPy)")

        # Read first frame
        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("Cannot read first frame")

        frame_num = 1
        last_scene_frame = 0
        self.scene_timestamps = [0.0]
        all_metrics = []

        print("\n" + "=" * 70)
        print("Detecting hard cuts (GPU accelerated)...")
        print("=" * 70)

        # Frame buffer for batch processing - start with first frame
        frame_buffer = [first_frame]
        frame_indices = [0]  # Track frame numbers for each buffered frame

        # Use async frame reading to overlap I/O with GPU computation
        for batch in read_frames_async(cap, batch_size=batch_size):
            # Add batch frames to buffer
            for frame in batch:
                frame_buffer.append(frame)
                frame_indices.append(frame_num)
                frame_num += 1

            # Process batch when buffer is full (batch_size + 1 for overlap)
            while len(frame_buffer) >= batch_size + 1:
                last_scene_frame = self._process_gpu_batch(
                    frame_buffer,
                    frame_indices,
                    fps,
                    min_frame_gap,
                    last_scene_frame,
                    all_metrics,
                    debug,
                    total_frames,
                    compute_pixel_difference_batch_gpu,
                    compute_histogram_distance_batch_gpu,
                    free_gpu_memory,
                    _stack_frames_to_gpu,  # Phase 2: single upload
                )

                # Keep last frame for next batch overlap
                frame_buffer = [frame_buffer[-1]]
                frame_indices = [frame_indices[-1]]

        # Process remaining frames in buffer
        if len(frame_buffer) > 1:
            self._process_gpu_batch(
                frame_buffer,
                frame_indices,
                fps,
                min_frame_gap,
                last_scene_frame,
                all_metrics,
                debug,
                total_frames,
                compute_pixel_difference_batch_gpu,
                compute_histogram_distance_batch_gpu,
                free_gpu_memory,
                _stack_frames_to_gpu,  # Phase 2: single upload
            )

        cap.release()

        self._print_detection_summary(duration, debug, all_metrics)
        return self.scene_timestamps

    def _determine_batch_size(self, frame_height: int, frame_width: int, debug: bool) -> int:
        """Determine the batch size to use for GPU processing."""
        if self._gpu_batch_size == "auto":
            # Auto-select based on GPU memory
            available_mb = self._gpu_info.memory_free_mb or 4096  # Default to 4GB if unknown
            batch_size = estimate_optimal_batch_size(
                frame_height, frame_width, available_mb, self._gpu_memory_fraction
            )
            if debug:
                print(f"🔧 Auto-selected batch size: {batch_size} (based on {available_mb}MB free)")
        else:
            batch_size = int(self._gpu_batch_size)
            # Clamp to valid range
            batch_size = max(5, min(batch_size, 120))

        return batch_size

    def _read_frames_sync(self, cap, batch_size: int):
        """
        Read frames synchronously in batches.

        This is used when async I/O provides no benefit (CPU mode) to avoid
        the ~10-20% overhead from thread management.

        Based on benchmark results (2026-01-09):
        - Async without GPU delay: 0.76-0.93x (10-20% slower)
        - Sync is preferred for pure CPU processing

        Args:
            cap: OpenCV VideoCapture object (must already be opened).
            batch_size: Number of frames per batch.

        Yields:
            list: Batch of frames (numpy arrays).
        """
        batch = []
        while True:
            ret, frame = cap.read()
            if not ret:
                if batch:
                    yield batch
                break

            batch.append(frame)
            if len(batch) >= batch_size:
                yield batch
                batch = []

    def _process_gpu_batch(
        self,
        frame_buffer,
        frame_indices,
        fps,
        min_frame_gap,
        last_scene_frame,
        all_metrics,
        debug,
        total_frames,
        compute_pixel_diff_fn,
        compute_hist_dist_fn,
        free_memory_fn,
        stack_frames_fn=None,
    ):
        """
        Process a batch of frames on GPU with OOM recovery.

        Returns the updated last_scene_frame value.

        Phase 2 optimization: Upload frames to GPU once and pass the GPU array to
        both metric computations. The metric functions will short-circuit when
        receiving a CuPy array (no re-stack or re-upload).

        Phase 3 optimization: GPU memory is managed by CuPy's memory pool without
        per-batch flushes. free_gpu_memory() is only called during error recovery
        (OOM/GPU errors) before falling back to CPU processing.

        When debug=True, timing information is recorded for:
        - CPU stack + GPU upload time (single upload)
        - Pixel difference computation time
        - Histogram distance computation time
        """
        if len(frame_buffer) < 2:
            return last_scene_frame

        try:
            batch_start = time.perf_counter() if debug else 0

            # Phase 2: Upload once, use twice
            # Stack frames and upload to GPU once
            if stack_frames_fn is not None:
                frames_gpu, stack_timing = stack_frames_fn(frame_buffer, debug=debug)
                stack_ms = stack_timing.get("cpu_stack_ms", 0) if debug else 0
                gpu_upload_ms = stack_timing.get("gpu_upload_ms", 0) if debug else 0
            else:
                # Fallback: pass frame_buffer directly (each function will upload)
                frames_gpu = frame_buffer
                stack_ms = 0
                gpu_upload_ms = 0

            # Compute pixel differences (short-circuits if frames_gpu is CuPy array)
            pixel_start = time.perf_counter() if debug else 0
            mean_diffs, changed_ratios = compute_pixel_diff_fn(frames_gpu, debug=debug)
            pixel_time_ms = (time.perf_counter() - pixel_start) * 1000 if debug else 0

            # Compute histogram distances (short-circuits if frames_gpu is CuPy array)
            hist_start = time.perf_counter() if debug else 0
            hist_distances = compute_hist_dist_fn(frames_gpu, debug=debug)
            hist_time_ms = (time.perf_counter() - hist_start) * 1000 if debug else 0

            # Phase 3: Removed unconditional free_gpu_memory() per batch
            # GPU memory is now managed by CuPy's memory pool (no per-batch flush)
            # free_gpu_memory() is only called during error recovery scenarios

            batch_time_ms = (time.perf_counter() - batch_start) * 1000 if debug else 0

            if debug:
                print(
                    f"\n  [GPU Batch {len(frame_buffer)} frames] "
                    f"stack={stack_ms:.1f}ms, upload={gpu_upload_ms:.1f}ms, "
                    f"pixel_diff={pixel_time_ms:.1f}ms, histogram={hist_time_ms:.1f}ms, "
                    f"total={batch_time_ms:.1f}ms"
                )

        except Exception as e:
            # Handle OOM or other GPU errors - fall back to CPU for this batch
            if "memory" in str(e).lower() or "OutOfMemory" in str(type(e).__name__):
                print(f"\n⚠ GPU memory error, processing batch on CPU: {e}")
            else:
                print(f"\n⚠ GPU error, processing batch on CPU: {e}")

            # Phase 3: Free GPU memory only during error recovery before CPU fallback
            free_memory_fn(debug=debug)

            # Fall back to CPU processing for this batch
            return self._process_cpu_batch_fallback(
                frame_buffer,
                frame_indices,
                fps,
                min_frame_gap,
                last_scene_frame,
                all_metrics,
                debug,
                total_frames,
            )

        # Process results (on CPU)
        for i in range(len(mean_diffs)):
            frame_idx = frame_indices[i + 1]  # Frame index of the second frame in pair
            hist_dist = float(hist_distances[i])
            pixel_diff = float(mean_diffs[i])
            changed_ratio = float(changed_ratios[i])

            metrics = {
                "frame": frame_idx,
                "timestamp": frame_idx / fps,
                "hist_dist": hist_dist,
                "pixel_diff": pixel_diff,
                "changed_ratio": changed_ratio * 100,
            }
            all_metrics.append(metrics)

            # Check for hard cut
            frames_since_last = frame_idx - last_scene_frame

            if is_hard_cut(
                hist_dist,
                pixel_diff,
                changed_ratio,
                self.threshold,
                frames_since_last,
                min_frame_gap,
            ):
                timestamp = frame_idx / fps
                self.scene_timestamps.append(timestamp)

                print(
                    f"✓ Scene {len(self.scene_timestamps)} at {timestamp:.2f}s (frame {frame_idx})"
                )
                print(
                    f"  Hist: {hist_dist:.1f}, Pixel: {pixel_diff:.1f}, "
                    f"Changed: {changed_ratio * 100:.1f}%"
                )

                last_scene_frame = frame_idx

                # Save debug frames if requested
                if debug:
                    save_debug_frames(
                        self.output_dir,
                        len(self.scene_timestamps),
                        frame_idx - 1,
                        frame_idx,
                        frame_buffer[i],
                        frame_buffer[i + 1],
                    )

            # Progress indicator
            if frame_idx % 100 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"Progress: {progress:.1f}%", end="\r")

        return last_scene_frame

    def _process_cpu_batch_fallback(
        self,
        frame_buffer,
        frame_indices,
        fps,
        min_frame_gap,
        last_scene_frame,
        all_metrics,
        debug,
        total_frames,
    ):
        """Process a batch of frames on CPU as fallback when GPU fails."""
        for i in range(len(frame_buffer) - 1):
            prev_frame = frame_buffer[i]
            curr_frame = frame_buffer[i + 1]
            frame_idx = frame_indices[i + 1]

            # Use CPU detection functions
            hist_dist = compute_histogram_distance(prev_frame, curr_frame)
            pixel_diff, changed_ratio = compute_pixel_difference(prev_frame, curr_frame)

            metrics = {
                "frame": frame_idx,
                "timestamp": frame_idx / fps,
                "hist_dist": hist_dist,
                "pixel_diff": pixel_diff,
                "changed_ratio": changed_ratio * 100,
            }
            all_metrics.append(metrics)

            frames_since_last = frame_idx - last_scene_frame

            if is_hard_cut(
                hist_dist,
                pixel_diff,
                changed_ratio,
                self.threshold,
                frames_since_last,
                min_frame_gap,
            ):
                timestamp = frame_idx / fps
                self.scene_timestamps.append(timestamp)

                print(
                    f"✓ Scene {len(self.scene_timestamps)} at {timestamp:.2f}s (frame {frame_idx})"
                )
                print(
                    f"  Hist: {hist_dist:.1f}, Pixel: {pixel_diff:.1f}, "
                    f"Changed: {changed_ratio * 100:.1f}%"
                )

                last_scene_frame = frame_idx

                if debug:
                    save_debug_frames(
                        self.output_dir,
                        len(self.scene_timestamps),
                        frame_idx - 1,
                        frame_idx,
                        prev_frame,
                        curr_frame,
                    )

            if frame_idx % 100 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"Progress: {progress:.1f}%", end="\r")

        return last_scene_frame

    def _process_hybrid_batch(
        self,
        frame_buffer,
        frame_indices,
        fps,
        min_frame_gap,
        last_scene_frame,
        all_metrics,
        debug,
        total_frames,
        hist_processor,
        pixel_processor,
        compute_pixel_diff_gpu_fn,
        free_memory_fn,
    ):
        """
        Process a batch of frames using hybrid CPU/GPU processing.

        Based on Phase 2C benchmark results:
        - Histogram: Always CPU (1.35x faster than GPU)
        - Pixel diff: GPU for HD+ (1.5-2x faster), CPU for SD

        Phase 3 optimization: GPU memory is managed by CuPy's memory pool without
        per-batch flushes. free_gpu_memory() is only called during error recovery
        (OOM/GPU errors) before falling back to CPU processing.

        When debug=True, timing information is recorded for:
        - CPU histogram computation
        - GPU/CPU pixel difference computation

        Returns the updated last_scene_frame value.
        """
        # Import batch CPU functions locally to avoid import issues
        from .detection import (
            compute_histogram_distance_batch_cpu,
            compute_pixel_difference_batch_cpu,
        )

        if len(frame_buffer) < 2:
            return last_scene_frame

        batch_start = time.perf_counter() if debug else 0

        # Histogram: Always CPU (based on benchmarks)
        hist_start = time.perf_counter() if debug else 0
        hist_distances = compute_histogram_distance_batch_cpu(frame_buffer)
        hist_time_ms = (time.perf_counter() - hist_start) * 1000 if debug else 0

        # Pixel difference: GPU or CPU based on resolution
        pixel_start = time.perf_counter() if debug else 0
        pixel_mode = "GPU" if pixel_processor == ProcessorType.GPU else "CPU"

        if pixel_processor == ProcessorType.GPU:
            try:
                mean_diffs, changed_ratios = compute_pixel_diff_gpu_fn(frame_buffer, debug=debug)
                pixel_time_ms = (time.perf_counter() - pixel_start) * 1000 if debug else 0
                # Phase 3: No unconditional free_gpu_memory() per batch
                # GPU memory managed by CuPy's memory pool
            except Exception as e:
                # Fall back to CPU on GPU error
                if "memory" in str(e).lower() or "OutOfMemory" in str(type(e).__name__):
                    print(f"\n⚠ GPU memory error, using CPU for pixel diff: {e}")
                else:
                    print(f"\n⚠ GPU error, using CPU for pixel diff: {e}")
                # Phase 3: Free GPU memory only during error recovery
                free_memory_fn(debug=debug)
                pixel_results = compute_pixel_difference_batch_cpu(frame_buffer)
                mean_diffs = [r[0] for r in pixel_results]
                changed_ratios = [r[1] for r in pixel_results]
                pixel_mode = "CPU (fallback)"
                pixel_time_ms = (time.perf_counter() - pixel_start) * 1000 if debug else 0
        else:
            # CPU processing
            pixel_results = compute_pixel_difference_batch_cpu(frame_buffer)
            mean_diffs = [r[0] for r in pixel_results]
            changed_ratios = [r[1] for r in pixel_results]
            pixel_time_ms = (time.perf_counter() - pixel_start) * 1000 if debug else 0

        batch_time_ms = (time.perf_counter() - batch_start) * 1000 if debug else 0

        if debug:
            print(
                f"\n  [Hybrid Batch {len(frame_buffer)} frames] "
                f"hist(CPU)={hist_time_ms:.1f}ms, pixel({pixel_mode})={pixel_time_ms:.1f}ms, "
                f"total={batch_time_ms:.1f}ms"
            )

        # Process results
        for i in range(len(mean_diffs)):
            frame_idx = frame_indices[i + 1]
            hist_dist = float(hist_distances[i])
            pixel_diff = float(mean_diffs[i])
            changed_ratio = float(changed_ratios[i])

            metrics = {
                "frame": frame_idx,
                "timestamp": frame_idx / fps,
                "hist_dist": hist_dist,
                "pixel_diff": pixel_diff,
                "changed_ratio": changed_ratio * 100,
            }
            all_metrics.append(metrics)

            frames_since_last = frame_idx - last_scene_frame

            if is_hard_cut(
                hist_dist,
                pixel_diff,
                changed_ratio,
                self.threshold,
                frames_since_last,
                min_frame_gap,
            ):
                timestamp = frame_idx / fps
                self.scene_timestamps.append(timestamp)

                print(
                    f"✓ Scene {len(self.scene_timestamps)} at {timestamp:.2f}s (frame {frame_idx})"
                )
                print(
                    f"  Hist: {hist_dist:.1f}, Pixel: {pixel_diff:.1f}, "
                    f"Changed: {changed_ratio * 100:.1f}%"
                )

                last_scene_frame = frame_idx

                if debug:
                    save_debug_frames(
                        self.output_dir,
                        len(self.scene_timestamps),
                        frame_idx - 1,
                        frame_idx,
                        frame_buffer[i],
                        frame_buffer[i + 1],
                    )

            if frame_idx % 100 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"Progress: {progress:.1f}%", end="\r")

        return last_scene_frame

    def _print_detection_summary(self, duration, debug, all_metrics):
        """Print the detection summary and save metrics if in debug mode."""
        print("\n" + "=" * 70)
        print(f"✓ Detected {len(self.scene_timestamps)} scenes")

        # Print scene durations
        if len(self.scene_timestamps) > 1:
            print("\nScene durations:")
            for i in range(len(self.scene_timestamps) - 1):
                dur = self.scene_timestamps[i + 1] - self.scene_timestamps[i]
                print(f"  Scene {i + 1}: {dur:.2f}s")

            last_dur = duration - self.scene_timestamps[-1]
            print(f"  Scene {len(self.scene_timestamps)}: {last_dur:.2f}s")

        # Print GPU timing summary if in debug mode and GPU was used
        processor_str = self._processor_request.value if hasattr(self, "_processor_request") else ""
        if debug and processor_str in ("gpu", "auto"):
            try:
                from .detection_gpu import get_accumulated_timings, reset_accumulated_timings

                timings = get_accumulated_timings()
                summary = timings.get_summary()

                if summary["batch_count"] > 0:
                    print("\n" + "-" * 50)
                    print("GPU Timing Summary:")
                    print(f"  Batches processed: {summary['batch_count']}")
                    print(f"  Total GPU time: {summary['total_ms']:.1f}ms")
                    print(f"  Avg batch time: {summary['avg_batch_ms']:.1f}ms")
                    print(f"  CPU stack time: {summary['cpu_stack_ms']:.1f}ms")
                    print(f"  GPU upload time: {summary['gpu_upload_ms']:.1f}ms")
                    print(f"  GPU download time: {summary['gpu_download_ms']:.1f}ms")
                    print(f"  Pixel diff compute: {summary['pixel_diff_compute_ms']:.1f}ms")
                    print(f"  Histogram compute: {summary['histogram_compute_ms']:.1f}ms")

                    # Calculate transfer overhead percentage
                    transfer_time = (
                        summary["cpu_stack_ms"]
                        + summary["gpu_upload_ms"]
                        + summary["gpu_download_ms"]
                    )
                    if summary["total_ms"] > 0:
                        transfer_pct = (transfer_time / summary["total_ms"]) * 100
                        print(f"  Transfer overhead: {transfer_pct:.1f}%")

                # Reset timings for next run
                reset_accumulated_timings()
            except ImportError:
                pass  # GPU module not available

        # Save analysis data if debug mode is enabled
        if debug:
            save_metrics_to_csv(all_metrics, self.output_dir)

    def save_timestamps(self, filename="timestamps.txt"):
        """
        Save detected scene timestamps to a text file.

        Creates a human-readable text file containing all detected scene start
        times along with metadata about the video and detection parameters.
        Useful for reference, manual review, or importing into other tools.

        Args:
            filename (str, optional): Name of the output file.
                Defaults to "timestamps.txt". The file is saved in output_dir.

        Output Format:
            Video: {video_path}
            Total scenes: {count}
            Threshold: {threshold}

            Scene 1: 0.00s
            Scene 2: 5.23s
            Scene 3: 12.45s
            ...

        Side Effects:
            - Creates a text file in output_dir
            - Prints confirmation message with file path

        Example:
            >>> splitter.save_timestamps("my_scenes.txt")
            ✓ Timestamps saved to output/my_scenes.txt
        """
        save_timestamps_to_file(
            self.video_path, self.scene_timestamps, self.threshold, self.output_dir, filename
        )

    def split_video(self):
        """
        Split video into separate files at detected scene boundaries.

        This method uses FFmpeg to split the original video into multiple files,
        one for each detected scene. The splitting is frame-accurate, achieved by
        re-encoding the video with H.264 codec. Each output file contains one
        complete scene from start to end (or to the next scene boundary).

        Encoder selection is based on the processor mode:
        - "cpu": Uses libx264 software encoding
        - "gpu": Uses NVENC hardware encoding (requires NVIDIA GPU)
        - "auto": Uses NVENC if available, otherwise falls back to libx264

        Returns:
            int: Number of successfully created video files.

        Example:
            >>> splitter.detect_scenes()
            >>> count = splitter.split_video()
            Splitting video with frame-accurate precision...
            Encoder: h264_nvenc (hardware)
            Scene 001: 0.00s → 5.23s
            Scene 002: 5.23s → 12.45s
            ✓ Created 2 video files in 'output'
        """
        return split_video_at_timestamps(
            self.video_path, self.scene_timestamps, self.output_dir, self._processor_request
        )
