"""
motion.py - Tumblin' Dice Motion Detector & Quiescence State Machine

This module provides:
1. Frame-differencing motion detector using OpenCV (cv2.absdiff, thresholding,
   morphological filtering, and multi-frame accumulation).
2. Quiescence timer tracking continuous stillness (>1.2s stillness by default)
   to freeze the scoreboard while dice tumble and automatically trigger scoring
   on settle.
3. Motion state machine (IDLE, MOVING, SETTLING, SETTLED) with scoreboard freeze
   gating and edge-triggered settle events.
4. Support for rectangular and arbitrary polygonal Regions of Interest (ROI)
   and masks (e.g., scoring zones or rectified board contours).
5. Headless/deterministic time injection support for high-speed unit testing
   and reproducible video replay.
6. HUD visual overlay rendering with status badges, quiescence progress bars,
   motion bounding boxes, and score meters.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from enum import Enum
import logging
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Default configuration constants
DEFAULT_QUIESCENCE_DURATION: float = 1.2   # Seconds of continuous stillness required
DEFAULT_MOTION_THRESHOLD: float = 500.0    # Movement score threshold (changed pixels)
DEFAULT_DIFF_THRESHOLD: int = 25           # Intensity difference threshold for binary mask
DEFAULT_BLUR_KSIZE: Tuple[int, int] = (21, 21)  # Gaussian blur kernel size
DEFAULT_MIN_CONTOUR_AREA: float = 50.0     # Minimum contour area to treat as real motion
DEFAULT_ACCUMULATION_FRAMES: int = 3       # Number of difference masks in rolling buffer
DEFAULT_DILATION_ITERATIONS: int = 2       # Morphological dilation iterations to bridge gaps


class MotionState(str, Enum):
    """
    Finite state machine states for board motion and quiescence gating.
    """
    IDLE = "idle"            # Board is quiescent; ready for next flick / roll
    MOVING = "moving"        # Active motion detected (dice tumbling or player flicking) -> "Rolling..."
    SETTLING = "settling"    # Movement dropped below threshold; accumulating stillness towards quiescence
    SETTLED = "settled"      # Quiescence duration reached (>1.2s stillness); triggers scoring pass


@dataclass
class MotionConfig:
    """
    Configuration parameters for frame-differencing motion detection and quiescence gating.
    """
    quiescence_duration: float = DEFAULT_QUIESCENCE_DURATION
    motion_threshold: float = DEFAULT_MOTION_THRESHOLD
    diff_threshold: int = DEFAULT_DIFF_THRESHOLD
    blur_ksize: Tuple[int, int] = DEFAULT_BLUR_KSIZE
    min_contour_area: float = DEFAULT_MIN_CONTOUR_AREA
    accumulation_frames: int = DEFAULT_ACCUMULATION_FRAMES
    dilation_iterations: int = DEFAULT_DILATION_ITERATIONS
    gaussian_sigma: float = 0.0
    roi: Optional[Tuple[int, int, int, int]] = None  # Rectangular ROI: (x, y, w, h)
    roi_polygon: Optional[List[Tuple[float, float]]] = None  # Polygon ROI vertices

    def __post_init__(self) -> None:
        if self.quiescence_duration <= 0.0:
            raise ValueError(f"quiescence_duration must be positive, got {self.quiescence_duration}")
        if self.motion_threshold < 0.0:
            raise ValueError(f"motion_threshold cannot be negative, got {self.motion_threshold}")
        if not (0 <= self.diff_threshold <= 255):
            raise ValueError(f"diff_threshold must be between 0 and 255, got {self.diff_threshold}")
        if len(self.blur_ksize) != 2 or self.blur_ksize[0] % 2 == 0 or self.blur_ksize[1] % 2 == 0:
            raise ValueError(f"blur_ksize must be a tuple of two odd positive integers, got {self.blur_ksize}")
        if self.accumulation_frames < 1:
            raise ValueError(f"accumulation_frames must be >= 1, got {self.accumulation_frames}")
        if self.dilation_iterations < 0:
            raise ValueError(f"dilation_iterations cannot be negative, got {self.dilation_iterations}")
        if self.roi is not None:
            if len(self.roi) != 4 or self.roi[2] <= 0 or self.roi[3] <= 0:
                raise ValueError(f"roi must be (x, y, w, h) with w, h > 0, got {self.roi}")
        if self.roi_polygon is not None:
            if len(self.roi_polygon) < 3:
                raise ValueError(f"roi_polygon must contain at least 3 vertices, got {len(self.roi_polygon)}")

    def to_dict(self) -> Dict[str, Any]:
        """Serializes configuration to dictionary."""
        return {
            "quiescence_duration": float(self.quiescence_duration),
            "motion_threshold": float(self.motion_threshold),
            "diff_threshold": int(self.diff_threshold),
            "blur_ksize": list(self.blur_ksize),
            "min_contour_area": float(self.min_contour_area),
            "accumulation_frames": int(self.accumulation_frames),
            "dilation_iterations": int(self.dilation_iterations),
            "gaussian_sigma": float(self.gaussian_sigma),
            "roi": list(self.roi) if self.roi is not None else None,
            "roi_polygon": [[float(p[0]), float(p[1])] for p in self.roi_polygon] if self.roi_polygon is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MotionConfig:
        """Deserializes configuration from dictionary."""
        blur = tuple(data.get("blur_ksize", DEFAULT_BLUR_KSIZE))
        roi = tuple(data["roi"]) if data.get("roi") is not None else None
        poly = [tuple(p) for p in data["roi_polygon"]] if data.get("roi_polygon") is not None else None
        return cls(
            quiescence_duration=float(data.get("quiescence_duration", DEFAULT_QUIESCENCE_DURATION)),
            motion_threshold=float(data.get("motion_threshold", DEFAULT_MOTION_THRESHOLD)),
            diff_threshold=int(data.get("diff_threshold", DEFAULT_DIFF_THRESHOLD)),
            blur_ksize=blur,  # type: ignore[arg-type]
            min_contour_area=float(data.get("min_contour_area", DEFAULT_MIN_CONTOUR_AREA)),
            accumulation_frames=int(data.get("accumulation_frames", DEFAULT_ACCUMULATION_FRAMES)),
            dilation_iterations=int(data.get("dilation_iterations", DEFAULT_DILATION_ITERATIONS)),
            gaussian_sigma=float(data.get("gaussian_sigma", 0.0)),
            roi=roi,  # type: ignore[arg-type]
            roi_polygon=poly,  # type: ignore[arg-type]
        )


@dataclass
class MotionResult:
    """
    Result container produced for each processed video frame.
    """
    state: MotionState
    movement_score: float
    motion_detected: bool
    is_settled: bool
    is_scoreboard_frozen: bool
    just_settled: bool
    stillness_duration: float
    quiescence_progress: float
    timestamp: float
    motion_contours: List[np.ndarray] = field(default_factory=list)
    bounding_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    motion_mask: Optional[np.ndarray] = None
    accumulated_mask: Optional[np.ndarray] = None
    accumulated_score: float = 0.0
    delta_frame: Optional[np.ndarray] = None

    @property
    def is_moving(self) -> bool:
        """Returns True if state is MOVING."""
        return self.state == MotionState.MOVING

    @property
    def is_settling(self) -> bool:
        """Returns True if state is SETTLING."""
        return self.state == MotionState.SETTLING


class MotionDetector:
    """
    Real-time frame-differencing motion detector and quiescence state machine.

    Tracks whether dice are actively tumbling or players are interacting with the board,
    freezes the live scoreboard while motion is active or settling, and triggers CV
    scoring passes once continuous stillness exceeds the quiescence timer (>1.2s).
    """

    def __init__(
        self,
        config: Optional[MotionConfig] = None,
        quiescence_duration: Optional[float] = None,
        motion_threshold: Optional[float] = None,
        diff_threshold: Optional[int] = None,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = MotionConfig()

        # Allow targeted overrides if specified
        if quiescence_duration is not None:
            self.config.quiescence_duration = float(quiescence_duration)
        if motion_threshold is not None:
            self.config.motion_threshold = float(motion_threshold)
        if diff_threshold is not None:
            self.config.diff_threshold = int(diff_threshold)

        # State tracking
        self._state: MotionState = MotionState.IDLE
        self._previous_state: MotionState = MotionState.IDLE
        self._is_scoreboard_frozen: bool = False
        self._just_settled: bool = False

        # Timing tracking
        self._last_timestamp: Optional[float] = None
        self._stillness_start_time: Optional[float] = None
        self._motion_start_time: Optional[float] = None
        self._stillness_duration: float = 0.0

        # Frame buffers
        self._prev_blurred: Optional[np.ndarray] = None
        self._history_masks: collections.deque[np.ndarray] = collections.deque(
            maxlen=self.config.accumulation_frames
        )
        self._last_result: Optional[MotionResult] = None

        # ROI & Mask caching
        self._roi_mask: Optional[np.ndarray] = None
        self._cached_frame_shape: Optional[Tuple[int, int]] = None

        # Callbacks
        self._on_settle_callbacks: List[Callable[[MotionResult], None]] = []
        self._on_motion_start_callbacks: List[Callable[[MotionResult], None]] = []

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def state(self) -> MotionState:
        """Current motion state."""
        return self._state

    @property
    def is_scoreboard_frozen(self) -> bool:
        """
        True when scoreboard must freeze: during active motion ('Rolling...')
        or while waiting for quiescence timer to confirm the settle.
        """
        return self._is_scoreboard_frozen

    @property
    def is_moving(self) -> bool:
        """True if scene is currently in active motion."""
        return self._state == MotionState.MOVING

    @property
    def is_settling(self) -> bool:
        """True if motion ceased but quiescence timer has not yet reached threshold."""
        return self._state == MotionState.SETTLING

    @property
    def is_settled(self) -> bool:
        """True if scene is settled (quiescence completed or idle)."""
        return self._state in (MotionState.SETTLED, MotionState.IDLE)

    @property
    def just_settled(self) -> bool:
        """Edge-triggered flag: True ONLY on the single frame where quiescence is reached."""
        return self._just_settled

    @property
    def stillness_duration(self) -> float:
        """Continuous seconds of stillness accumulated in current quiescent period."""
        return self._stillness_duration

    @property
    def quiescence_progress(self) -> float:
        """Progress towards settle threshold (0.0 to 1.0)."""
        if self._state == MotionState.SETTLED:
            return 1.0
        if self._state == MotionState.MOVING or self.config.quiescence_duration <= 0:
            return 0.0
        return min(1.0, max(0.0, self._stillness_duration / self.config.quiescence_duration))

    @property
    def movement_score(self) -> float:
        """Most recent frame movement score (count of moving pixels)."""
        return self._last_result.movement_score if self._last_result is not None else 0.0

    @property
    def last_result(self) -> Optional[MotionResult]:
        """Most recent MotionResult instance."""
        return self._last_result

    # -------------------------------------------------------------------------
    # ROI & Mask Configuration
    # -------------------------------------------------------------------------

    def set_roi(self, x: int, y: int, w: int, h: int) -> None:
        """Configures an axis-aligned rectangular Region of Interest."""
        if w <= 0 or h <= 0:
            raise ValueError(f"ROI width and height must be positive, got ({w}, {h})")
        self.config.roi = (int(x), int(y), int(w), int(h))
        self.config.roi_polygon = None
        self._roi_mask = None

    def set_roi_polygon(self, points: Sequence[Sequence[float]]) -> None:
        """Configures an arbitrary polygon Region of Interest."""
        pts = [(float(p[0]), float(p[1])) for p in points]
        if len(pts) < 3:
            raise ValueError(f"Polygon ROI requires at least 3 points, got {len(pts)}")
        self.config.roi_polygon = pts
        self.config.roi = None
        self._roi_mask = None

    def set_mask(self, mask: np.ndarray) -> None:
        """Directly sets a custom binary mask (255 = inspect, 0 = ignore)."""
        if mask is None or len(mask.shape) != 2:
            raise ValueError("Custom mask must be a 2D grayscale/binary numpy array")
        self._roi_mask = (mask > 0).astype(np.uint8) * 255
        self._cached_frame_shape = mask.shape[:2]

    def clear_roi(self) -> None:
        """Clears all ROI and polygon masks, inspecting the entire frame."""
        self.config.roi = None
        self.config.roi_polygon = None
        self._roi_mask = None

    def _ensure_roi_mask(self, height: int, width: int) -> Optional[np.ndarray]:
        """Builds or returns cached binary ROI mask matching frame dimensions."""
        if self._roi_mask is not None and self._cached_frame_shape == (height, width):
            return self._roi_mask

        if self.config.roi_polygon is not None:
            mask = np.zeros((height, width), dtype=np.uint8)
            pts = np.array(self.config.roi_polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)
            self._roi_mask = mask
            self._cached_frame_shape = (height, width)
            return self._roi_mask

        if self.config.roi is not None:
            mask = np.zeros((height, width), dtype=np.uint8)
            rx, ry, rw, rh = self.config.roi
            # Clamp to frame bounds
            x1 = max(0, min(width, rx))
            y1 = max(0, min(height, ry))
            x2 = max(0, min(width, rx + rw))
            y2 = max(0, min(height, ry + rh))
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 255
            self._roi_mask = mask
            self._cached_frame_shape = (height, width)
            return self._roi_mask

        self._roi_mask = None
        self._cached_frame_shape = (height, width)
        return None

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def register_on_settle(self, callback: Callable[[MotionResult], None]) -> None:
        """Registers a callback invoked when motion settles (>1.2s stillness)."""
        if not callable(callback):
            raise TypeError("Callback must be callable")
        self._on_settle_callbacks.append(callback)

    def register_on_motion_start(self, callback: Callable[[MotionResult], None]) -> None:
        """Registers a callback invoked when motion first begins (rolling starts)."""
        if not callable(callback):
            raise TypeError("Callback must be callable")
        self._on_motion_start_callbacks.append(callback)

    # -------------------------------------------------------------------------
    # Failsafes and Manual Overrides
    # -------------------------------------------------------------------------

    def force_settle(self, timestamp: Optional[float] = None) -> MotionResult:
        """
        Failsafe override: immediately forces state to SETTLED, unfreezes scoreboard,
        and triggers settle callbacks. Used for UI 'Next Turn / Lock Frame' button.
        """
        now = float(time.monotonic() if timestamp is None else timestamp)
        self._previous_state = self._state
        self._state = MotionState.SETTLED
        self._is_scoreboard_frozen = False
        self._just_settled = True
        self._stillness_duration = self.config.quiescence_duration
        self._stillness_start_time = now - self.config.quiescence_duration

        result = MotionResult(
            state=self._state,
            movement_score=0.0,
            motion_detected=False,
            is_settled=True,
            is_scoreboard_frozen=False,
            just_settled=True,
            stillness_duration=self._stillness_duration,
            quiescence_progress=1.0,
            timestamp=now,
            motion_contours=[],
            bounding_boxes=[],
            motion_mask=None,
            delta_frame=None,
        )
        self._last_result = result
        self._trigger_settle_callbacks(result)
        return result

    def force_motion(self, timestamp: Optional[float] = None) -> MotionResult:
        """
        Failsafe override: forces state to MOVING, freezing the scoreboard.
        """
        now = float(time.monotonic() if timestamp is None else timestamp)
        self._previous_state = self._state
        self._state = MotionState.MOVING
        self._is_scoreboard_frozen = True
        self._just_settled = False
        self._stillness_duration = 0.0
        self._stillness_start_time = None
        self._motion_start_time = now

        result = MotionResult(
            state=self._state,
            movement_score=self.config.motion_threshold + 100.0,
            motion_detected=True,
            is_settled=False,
            is_scoreboard_frozen=True,
            just_settled=False,
            stillness_duration=0.0,
            quiescence_progress=0.0,
            timestamp=now,
            motion_contours=[],
            bounding_boxes=[],
            motion_mask=None,
            delta_frame=None,
        )
        self._last_result = result
        self._trigger_motion_start_callbacks(result)
        return result

    def reset(self) -> None:
        """Clears all historical buffers, resets state machine to IDLE, and unfreezes."""
        self._state = MotionState.IDLE
        self._previous_state = MotionState.IDLE
        self._is_scoreboard_frozen = False
        self._just_settled = False
        self._last_timestamp = None
        self._stillness_start_time = None
        self._motion_start_time = None
        self._stillness_duration = 0.0
        self._prev_blurred = None
        self._history_masks.clear()
        self._last_result = None

    # -------------------------------------------------------------------------
    # Core Processing Loop
    # -------------------------------------------------------------------------

    def update(self, frame: np.ndarray, timestamp: Optional[float] = None) -> MotionResult:
        """
        Processes a single video frame.

        Args:
            frame: Input image array (BGR, BGRA, or Grayscale).
            timestamp: Optional wall-clock or synthetic timestamp (seconds).
                       If None, time.monotonic() is used.

        Returns:
            MotionResult detailing motion state, scoreboard freeze flag, settle trigger, etc.
        """
        if frame is None:
            raise ValueError("Input frame cannot be None")
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Input frame must be a non-empty numpy ndarray")

        now = float(time.monotonic() if timestamp is None else timestamp)
        h, w = frame.shape[:2]

        # 1. Convert frame to grayscale
        if frame.ndim == 2:
            gray = frame.copy()
        elif frame.ndim == 3:
            if frame.shape[2] == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            elif frame.shape[2] == 4:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            elif frame.shape[2] == 1:
                gray = frame[:, :, 0].copy()
            else:
                raise ValueError(f"Unsupported number of image channels: {frame.shape[2]}")
        else:
            raise ValueError(f"Invalid frame dimensions: {frame.shape}")

        # 2. Gaussian blur to suppress camera sensor jitter / noise
        blurred = cv2.GaussianBlur(gray, self.config.blur_ksize, self.config.gaussian_sigma)

        # Handle first frame initialization or resolution changes
        if self._prev_blurred is None or self._prev_blurred.shape != blurred.shape:
            self._prev_blurred = blurred
            self._last_timestamp = now
            self._history_masks.clear()
            self._just_settled = False

            result = MotionResult(
                state=self._state,
                movement_score=0.0,
                motion_detected=False,
                is_settled=self._state in (MotionState.SETTLED, MotionState.IDLE),
                is_scoreboard_frozen=self._is_scoreboard_frozen,
                just_settled=False,
                stillness_duration=0.0,
                quiescence_progress=0.0,
                timestamp=now,
                motion_contours=[],
                bounding_boxes=[],
                motion_mask=np.zeros((h, w), dtype=np.uint8),
                delta_frame=np.zeros((h, w), dtype=np.uint8),
            )
            self._last_result = result
            return result

        # 3. Compute absolute frame difference
        delta = cv2.absdiff(self._prev_blurred, blurred)

        # 4. Threshold to binary motion mask
        _, thresh = cv2.threshold(delta, self.config.diff_threshold, 255, cv2.THRESH_BINARY)

        # 5. Morphological dilation to bridge small fragmented gaps across dice pips / edges
        if self.config.dilation_iterations > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thresh = cv2.dilate(thresh, kernel, iterations=self.config.dilation_iterations)

        # 6. Apply ROI / polygon mask if configured
        roi_mask = self._ensure_roi_mask(h, w)
        if roi_mask is not None:
            thresh = cv2.bitwise_and(thresh, roi_mask)
            delta = cv2.bitwise_and(delta, roi_mask)

        # 7. Update multi-frame accumulation buffer
        # Stores recent binary masks in rolling buffer
        if self.config.accumulation_frames > 1:
            self._history_masks.append(thresh.copy())
            accumulated_mask = np.bitwise_or.reduce(list(self._history_masks))
            accumulated_score = float(cv2.countNonZero(accumulated_mask))
        else:
            accumulated_mask = thresh
            accumulated_score = float(cv2.countNonZero(thresh))

        # 8. Movement score: count of active moving pixels in current frame difference
        movement_score = float(cv2.countNonZero(thresh))

        # 9. Extract contours and bounding boxes of moving entities (dice / hands)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_contours: List[np.ndarray] = []
        bounding_boxes: List[Tuple[int, int, int, int]] = []
        for c in contours:
            area = float(cv2.contourArea(c))
            if area >= self.config.min_contour_area:
                motion_contours.append(c)
                x, y, bw, bh = cv2.boundingRect(c)
                bounding_boxes.append((int(x), int(y), int(bw), int(bh)))

        # Determine motion flag
        # Motion is detected if movement score exceeds threshold or meaningful contours exist
        motion_detected = (movement_score > self.config.motion_threshold)

        # 10. Advance State Machine & Quiescence Timing
        self._just_settled = False
        fire_motion_start = False
        fire_settle = False

        if motion_detected:
            # Active movement
            if self._state != MotionState.MOVING:
                fire_motion_start = True

            self._previous_state = self._state
            self._state = MotionState.MOVING
            self._is_scoreboard_frozen = True
            self._stillness_start_time = None
            self._stillness_duration = 0.0

            if self._motion_start_time is None:
                self._motion_start_time = now

        else:
            # Below motion threshold: scene is still
            if self._state == MotionState.MOVING:
                # Just stopped moving -> begin settling phase
                self._previous_state = MotionState.MOVING
                self._state = MotionState.SETTLING
                self._is_scoreboard_frozen = True  # Hold scoreboard frozen while settling
                self._stillness_start_time = now
                self._stillness_duration = 0.0
                self._motion_start_time = None

            elif self._state == MotionState.SETTLING:
                # Accumulating stillness
                if self._stillness_start_time is None:
                    self._stillness_start_time = now
                self._stillness_duration = max(0.0, now - self._stillness_start_time)

                # Check quiescence threshold (>1.2s stillness)
                if self._stillness_duration >= self.config.quiescence_duration:
                    self._previous_state = MotionState.SETTLING
                    self._state = MotionState.SETTLED
                    self._is_scoreboard_frozen = False  # Unfreeze scoreboard!
                    self._just_settled = True  # Trigger CV scoring pass!
                    fire_settle = True
                else:
                    self._is_scoreboard_frozen = True  # Keep scoreboard frozen

            elif self._state == MotionState.SETTLED:
                # Already settled; remain settled
                if self._stillness_start_time is not None:
                    self._stillness_duration = max(0.0, now - self._stillness_start_time)
                else:
                    self._stillness_duration = self.config.quiescence_duration
                self._is_scoreboard_frozen = False
                self._just_settled = False

            elif self._state == MotionState.IDLE:
                # Idle baseline before any flick
                self._is_scoreboard_frozen = False
                self._stillness_duration = 0.0
                self._just_settled = False

        # Store for next iteration
        self._prev_blurred = blurred
        self._last_timestamp = now

        result = MotionResult(
            state=self._state,
            movement_score=movement_score,
            motion_detected=motion_detected,
            is_settled=self._state in (MotionState.SETTLED, MotionState.IDLE),
            is_scoreboard_frozen=self._is_scoreboard_frozen,
            just_settled=self._just_settled,
            stillness_duration=self._stillness_duration,
            quiescence_progress=self.quiescence_progress,
            timestamp=now,
            motion_contours=motion_contours,
            bounding_boxes=bounding_boxes,
            motion_mask=thresh,
            accumulated_mask=accumulated_mask,
            accumulated_score=accumulated_score,
            delta_frame=delta,
        )
        self._last_result = result

        # Dispatch registered callbacks
        if fire_motion_start:
            self._trigger_motion_start_callbacks(result)
        if fire_settle:
            self._trigger_settle_callbacks(result)

        return result

    def process_frame(self, frame: np.ndarray, timestamp: Optional[float] = None) -> MotionResult:
        """Alias for update()."""
        return self.update(frame=frame, timestamp=timestamp)

    def _trigger_settle_callbacks(self, result: MotionResult) -> None:
        """Dispatches on_settle callbacks safely."""
        for cb in self._on_settle_callbacks:
            try:
                cb(result)
            except Exception as e:
                logger.error("Error in on_settle callback: %s", e)

    def _trigger_motion_start_callbacks(self, result: MotionResult) -> None:
        """Dispatches on_motion_start callbacks safely."""
        for cb in self._on_motion_start_callbacks:
            try:
                cb(result)
            except Exception as e:
                logger.error("Error in on_motion_start callback: %s", e)

    # -------------------------------------------------------------------------
    # HUD & Visualization Overlay
    # -------------------------------------------------------------------------

    def draw_overlay(
        self,
        frame: np.ndarray,
        result: Optional[MotionResult] = None,
        show_badge: bool = True,
        show_contours: bool = True,
        show_meter: bool = True,
        show_roi: bool = True,
    ) -> np.ndarray:
        """
        Renders HUD overlay onto a copy of the input frame.

        Visualizes:
        - Status badge:
            - 'ROLLING...' (Amber/Red, Freeze Scoreboard) during motion
            - 'SETTLING (0.8s / 1.2s)' with smooth progress bar
            - 'SETTLED / READY' (Green) when quiescence threshold is crossed
        - Bounding boxes and contours of moving dice / hands
        - Movement score meter and threshold line
        - ROI boundary (if configured)
        """
        if frame is None or frame.size == 0:
            return frame

        res = result if result is not None else self._last_result
        output = frame.copy()
        h, w = output.shape[:2]

        # Draw ROI outline if active
        if show_roi:
            if self.config.roi_polygon is not None and len(self.config.roi_polygon) >= 3:
                pts = np.array(self.config.roi_polygon, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(output, [pts], isClosed=True, color=(255, 200, 0), thickness=2)
            elif self.config.roi is not None:
                rx, ry, rw, rh = self.config.roi
                cv2.rectangle(output, (rx, ry), (rx + rw, ry + rh), (255, 200, 0), 2)

        # Draw bounding boxes around moving blobs
        if show_contours and res is not None:
            for bx, by, bw, bh in res.bounding_boxes:
                # Bounding box in bright cyan
                cv2.rectangle(output, (bx, by), (bx + bw, by + bh), (255, 255, 0), 2)
                # Small corner accent
                cv2.circle(output, (bx + bw // 2, by + bh // 2), 3, (0, 255, 255), -1)

        # Status badge rendering
        if show_badge and res is not None:
            badge_w, badge_h = 280, 52
            badge_x, badge_y = 20, 20

            # Determine badge styling based on motion state
            if res.state == MotionState.MOVING:
                bg_color = (30, 40, 220)       # Crimson Red
                border_color = (80, 90, 255)
                text = "ROLLING... (FROZEN)"
                subtext = f"Motion: {int(res.movement_score)} px"
            elif res.state == MotionState.SETTLING:
                bg_color = (0, 140, 255)       # Amber / Orange
                border_color = (0, 200, 255)
                progress_pct = int(res.quiescence_progress * 100)
                text = f"SETTLING... {res.stillness_duration:.1f}s / {self.config.quiescence_duration:.1f}s"
                subtext = f"Quiescence: {progress_pct}%"
            elif res.state == MotionState.SETTLED:
                bg_color = (40, 160, 40)       # Green
                border_color = (80, 220, 80)
                text = "SETTLED / READY"
                subtext = "Scoreboard Active"
            else:  # IDLE
                bg_color = (70, 70, 70)         # Neutral Gray
                border_color = (120, 120, 120)
                text = "IDLE (WAITING FOR FLICK)"
                subtext = f"Threshold: {int(self.config.motion_threshold)} px"

            # Create semi-transparent overlay for badge
            badge_overlay = output.copy()
            cv2.rectangle(
                badge_overlay,
                (badge_x, badge_y),
                (badge_x + badge_w, badge_y + badge_h),
                bg_color,
                thickness=-1,
            )
            cv2.rectangle(
                badge_overlay,
                (badge_x, badge_y),
                (badge_x + badge_w, badge_y + badge_h),
                border_color,
                thickness=2,
            )

            # Quiescence progress bar when settling
            if res.state == MotionState.SETTLING:
                bar_h = 6
                bar_y = badge_y + badge_h - bar_h - 4
                bar_w_max = badge_w - 20
                bar_fill = int(bar_w_max * res.quiescence_progress)
                # Background track
                cv2.rectangle(
                    badge_overlay,
                    (badge_x + 10, bar_y),
                    (badge_x + 10 + bar_w_max, bar_y + bar_h),
                    (50, 50, 50),
                    -1,
                )
                # Fill
                cv2.rectangle(
                    badge_overlay,
                    (badge_x + 10, bar_y),
                    (badge_x + 10 + bar_fill, bar_y + bar_h),
                    (0, 255, 255),
                    -1,
                )

            # Blend badge overlay
            cv2.addWeighted(badge_overlay, 0.85, output, 0.15, 0, output)

            # Draw badge text
            cv2.putText(
                output,
                text,
                (badge_x + 12, badge_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                output,
                subtext,
                (badge_x + 12, badge_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )

        # Movement score meter in bottom-left
        if show_meter and res is not None:
            meter_w, meter_h = 220, 24
            meter_x, meter_y = 20, h - 40
            if meter_y > 0 and meter_x + meter_w < w:
                meter_overlay = output.copy()
                cv2.rectangle(
                    meter_overlay,
                    (meter_x, meter_y),
                    (meter_x + meter_w, meter_y + meter_h),
                    (30, 30, 30),
                    -1,
                )
                # Normalized fill based on 2x threshold
                max_score = max(self.config.motion_threshold * 2.0, 1.0)
                ratio = min(1.0, res.movement_score / max_score)
                fill_w = int((meter_w - 4) * ratio)
                fill_color = (0, 165, 255) if res.motion_detected else (100, 200, 100)

                cv2.rectangle(
                    meter_overlay,
                    (meter_x + 2, meter_y + 2),
                    (meter_x + 2 + fill_w, meter_y + meter_h - 2),
                    fill_color,
                    -1,
                )
                # Threshold marker line
                thresh_x = int(meter_x + 2 + (meter_w - 4) * (self.config.motion_threshold / max_score))
                cv2.line(
                    meter_overlay,
                    (thresh_x, meter_y),
                    (thresh_x, meter_y + meter_h),
                    (0, 0, 255),
                    2,
                )
                cv2.addWeighted(meter_overlay, 0.8, output, 0.2, 0, output)

                cv2.putText(
                    output,
                    f"Diff: {int(res.movement_score)} / {int(self.config.motion_threshold)}",
                    (meter_x + 6, meter_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        return output


def run_camera(camera_idx: int = 0, config: Optional[MotionConfig] = None) -> None:
    """Run live motion detection and quiescence gating from webcam."""
    if config is None:
        config = MotionConfig()

    detector = MotionDetector(config)
    detector.register_on_settle(lambda r: print(f"*** [EVENT] SETTLED: Stillness exceeded {config.quiescence_duration:.1f}s -> Triggering scoring pass! ***"))
    detector.register_on_motion_start(lambda r: print("*** [EVENT] MOTION DETECTED: Scoreboard frozen while dice tumble! ***"))

    # On Windows, DirectShow (CAP_DSHOW) avoids MSMF grabFrame -1072875772 errors on webcams like Logitech C920
    cap = None
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(camera_idx)
    else:
        cap = cv2.VideoCapture(camera_idx)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera device index {camera_idx}.")
        print("Please check your camera connection or try another device index.")
        return

    # Request smooth 640x480 resolution for low-latency motion detection
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    window_name = f"Tumblin' Dice Motion Detector (Camera {camera_idx})"
    print(f"Starting live camera stream on device {camera_idx}...")
    print("Controls: [q] or [ESC] to quit | [r] to reset | [f] to force settle")

    consecutive_failures = 0
    max_consecutive_failures = 20

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    print(f"\n[ERROR] Unable to grab frames from camera {camera_idx} ({max_consecutive_failures} consecutive failures).")
                    print("Possible causes:")
                    print("  1. Another application (Zoom, Teams, Discord, OBS, or Windows Camera app) is currently using the camera.")
                    print("  2. Camera privacy settings in Windows are blocking desktop app access.")
                    break
                time.sleep(0.05)
                continue

            consecutive_failures = 0
            res = detector.update(frame)
            overlay = detector.draw_overlay(frame, res)

            # Draw bottom guide line
            cv2.putText(
                overlay,
                "Controls: [q/ESC] Quit | [r] Reset | [f] Force Settle",
                (20, overlay.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, overlay)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            elif key in (ord('r'), ord('R')):
                detector.reset()
                print("[INFO] Detector reset to IDLE.")
            elif key in (ord('f'), ord('F')):
                detector.force_settle()
                print("[INFO] Force settled manually.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_demo(config: Optional[MotionConfig] = None) -> None:
    """Run interactive simulated demo demonstrating tumbling dice and quiescence settle."""
    if config is None:
        config = MotionConfig()

    detector = MotionDetector(config)
    detector.register_on_settle(lambda r: print(f"*** [EVENT] SETTLED: Stillness exceeded {config.quiescence_duration:.1f}s -> Triggering scoring pass! ***"))
    detector.register_on_motion_start(lambda r: print("*** [EVENT] MOTION DETECTED: Scoreboard frozen while dice tumble! ***"))

    window_name = "Tumblin' Dice Motion Detector — Interactive Demo"
    width, height = 720, 480
    board_bg = np.zeros((height, width, 3), dtype=np.uint8)
    # Felt green board background with wooden border
    board_bg[:] = (40, 95, 45)
    cv2.rectangle(board_bg, (0, 0), (width, height), (30, 45, 80), 16)

    print("=" * 65)
    print("Tumblin' Dice Motion & Quiescence Gating Demo")
    print("=" * 65)
    print("Demonstrating:")
    print("  1. ROLLING (MOVING): Dice tumble -> scoreboard freezes")
    print("  2. SETTLING: Dice come to rest -> stillness accumulated (1.2s)")
    print("  3. SETTLED: Quiescence achieved -> triggers score calculation")
    print("\nControls:")
    print("  [Space] Trigger a new dice roll")
    print("  [q/ESC] Quit demo")
    print("=" * 65)

    class DieSimulation:
        def __init__(self) -> None:
            self.reset()

        def reset(self) -> None:
            self.active = True
            self.x = 60.0
            self.y = float(height // 2)
            self.vx = 7.5 + np.random.uniform(1.0, 4.0)
            self.vy = np.random.uniform(-3.5, 3.5)
            self.angle = 0.0
            self.v_rot = np.random.uniform(8.0, 16.0)
            self.friction = 0.965
            self.size = 42

        def update(self) -> bool:
            """Advance simulation physics. Returns True if still moving."""
            if not self.active:
                return False

            self.x += self.vx
            self.y += self.vy
            self.angle += self.v_rot
            self.vx *= self.friction
            self.vy *= self.friction
            self.v_rot *= self.friction

            # Bounce off edges
            margin = 35
            if self.x < margin:
                self.x = margin
                self.vx = -self.vx * 0.7
            elif self.x > width - margin:
                self.x = width - margin
                self.vx = -self.vx * 0.7

            if self.y < margin:
                self.y = margin
                self.vy = -self.vy * 0.7
            elif self.y > height - margin:
                self.y = height - margin
                self.vy = -self.vy * 0.7

            speed = np.hypot(self.vx, self.vy)
            if speed < 0.25:
                self.active = False
                return False
            return True

        def draw(self, frame: np.ndarray) -> None:
            cx, cy = int(self.x), int(self.y)
            half = self.size // 2
            # Draw rotated die
            rect = ((cx, cy), (self.size, self.size), self.angle)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.fillPoly(frame, [box], (235, 235, 240))
            cv2.polylines(frame, [box], True, (30, 30, 30), 2)
            # Center pip
            cv2.circle(frame, (cx, cy), 5, (20, 20, 180), -1)

    sim = DieSimulation()
    auto_restart_time = 0.0

    try:
        while True:
            t_now = time.time()
            frame = board_bg.copy()

            # Advance die physics
            is_die_moving = sim.update()
            sim.draw(frame)

            # Update motion detector
            res = detector.update(frame, timestamp=t_now)
            overlay = detector.draw_overlay(frame, res)

            # Auto-restart 2.5s after settling to demonstrate continuous play
            if res.state == MotionState.SETTLED:
                if auto_restart_time == 0.0:
                    auto_restart_time = t_now + 2.5
                elif t_now >= auto_restart_time:
                    sim.reset()
                    auto_restart_time = 0.0
            else:
                auto_restart_time = 0.0

            # Demo guide text
            cv2.putText(
                overlay,
                "Demo: [Space] New Roll | [q/ESC] Quit",
                (20, overlay.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, overlay)
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            elif key == ord(' '):
                sim.reset()
                print("[INFO] Triggered new simulated dice roll.")
    finally:
        cv2.destroyAllWindows()


def main() -> None:
    """CLI entry point for motion detection module."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Tumblin' Dice Motion Detector & Quiescence State Machine"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run interactive simulated demo demonstrating tumbling dice and settle gating",
    )
    parser.add_argument(
        "--camera",
        type=int,
        nargs="?",
        const=0,
        default=None,
        help="Run live motion detection using webcam device index (default: 0)",
    )
    parser.add_argument(
        "--quiescence",
        type=float,
        default=DEFAULT_QUIESCENCE_DURATION,
        help=f"Quiescence stillness duration in seconds (default: {DEFAULT_QUIESCENCE_DURATION})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_MOTION_THRESHOLD,
        help=f"Motion sensitivity threshold (default: {DEFAULT_MOTION_THRESHOLD})",
    )

    args = parser.parse_args()

    config = MotionConfig(
        quiescence_duration=args.quiescence,
        motion_threshold=args.threshold,
    )

    if args.camera is not None:
        run_camera(camera_idx=args.camera, config=config)
    elif args.demo:
        run_demo(config=config)
    else:
        # If neither is specified, default to running demo mode
        print("[INFO] No mode specified. Defaulting to interactive simulation demo (--demo).")
        run_demo(config=config)


if __name__ == "__main__":
    main()
