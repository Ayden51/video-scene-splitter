"""
Video Scene Splitter - Automatic hard cut detection and video segmentation.

This package provides intelligent video scene detection using computer vision techniques
including HSV histogram analysis, pixel difference comparison, and changed pixel ratio
analysis to identify hard cuts (abrupt scene changes) in video content.
"""

from .splitter import VideoSceneSplitter

__version__ = "0.1.0"
__all__ = ["VideoSceneSplitter"]
