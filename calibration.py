"""
calibration.py - Tumblin' Dice Vision Scoreboard Calibration Module

This module provides:
1. OpenCV ArUco 4-corner homography perspective warp (DICT_4X4_50 fiducials)
   to normalize overhead webcam feeds into a flat, standardized 2D board plane.
2. ScoringZone data model supporting irregular polygons and 1x..6x multipliers.
3. ZoneManager for spatial point-in-polygon queries (cv2.pointPolygonTest),
   semi-transparent HUD overlay rendering, and JSON persistence.
4. InteractiveZoneEditor supporting mouse-driven polygon vertex entry,
   keyboard multiplier switching (1x..6x), undo, delete, and file save/load.
5. CalibrationPipeline coordinating end-to-end perspective warping, zone mapping,
   and bidirectional camera <-> board coordinate transformations.
6. Synthetic ArUco board sheet generator for physical printing and unit testing.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Standard multipliers for Tumblin' Dice
VALID_MULTIPLIERS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)

# Default BGR color palette for 1x..6x multiplier zones
DEFAULT_ZONE_COLORS: Dict[int, Tuple[int, int, int]] = {
    1: (255, 140, 0),    # Azure / Sky Blue (BGR)
    2: (50, 205, 50),     # Lime Green (BGR)
    3: (30, 30, 220),     # Crimson Red (BGR)
    4: (0, 165, 255),     # Amber Orange (BGR)
    5: (211, 0, 148),     # Violet / Magenta (BGR)
    6: (0, 215, 255),     # Gold Yellow (BGR)
}
DEFAULT_FALLBACK_COLOR: Tuple[int, int, int] = (180, 180, 180)
DEFAULT_CONFIG_PATH: str = "zones_config.json"
DEFAULT_BOARD_WIDTH: int = 800
DEFAULT_BOARD_HEIGHT: int = 1200
DEFAULT_MARKER_IDS: List[int] = [0, 1, 2, 3]  # TL, TR, BR, BL


def order_points_clockwise(points: Sequence[Sequence[float]]) -> np.ndarray:
    """
    Orders 4 planar 2D points clockwise starting from Top-Left:
    [Top-Left, Top-Right, Bottom-Right, Bottom-Left].

    Uses polar angle sorting relative to centroid with minimum (x+y) as Top-Left anchor.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError(f"Expected 4 2D points with shape (4, 2), got {pts.shape}")

    # Check for duplicate / collinear points
    if np.allclose(pts[0], pts[1]) or np.allclose(pts[0], pts[2]):
        raise ValueError("Degenerate points: duplicate or collinear points provided")

    centroid = np.mean(pts, axis=0)
    # Compute angles around centroid
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    order = np.argsort(angles)
    pts_sorted = pts[order]

    # Find Top-Left (point minimizing x + y)
    sums = pts_sorted[:, 0] + pts_sorted[:, 1]
    tl_idx = int(np.argmin(sums))

    # Roll so Top-Left is at index 0
    ordered = np.roll(pts_sorted, -tl_idx, axis=0)

    # Verify clockwise order by cross product of vector(0->1) and vector(0->3)
    v1 = ordered[1] - ordered[0]
    v3 = ordered[3] - ordered[0]
    cross = v1[0] * v3[1] - v1[1] * v3[0]
    if cross < 0:  # Counter-clockwise -> swap indices 1 and 3
        ordered = ordered[[0, 3, 2, 1]]

    return ordered


@dataclass
class ScoringZone:
    """
    Represents an irregular polygon scoring zone with a multiplier (1x..6x).
    Coordinates are defined in the rectified board plane (0..width, 0..height).
    """
    name: str
    multiplier: int
    points: List[Tuple[float, float]]
    color: Optional[Tuple[int, int, int]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.multiplier, (int, np.integer)):
            raise TypeError(f"Multiplier must be an integer, got {type(self.multiplier).__name__}")
        if self.multiplier not in VALID_MULTIPLIERS and self.multiplier != 0:
            raise ValueError(
                f"Invalid multiplier {self.multiplier}. Must be in {VALID_MULTIPLIERS} (or 0 for foul zone)."
            )
        if not self.points or len(self.points) < 3:
            raise ValueError(
                f"A polygon requires at least 3 vertices, got {len(self.points) if self.points else 0}."
            )

        # Normalize points to list of float tuples
        self.points = [(float(p[0]), float(p[1])) for p in self.points]

        # Assign default color if none provided
        if self.color is None:
            self.color = DEFAULT_ZONE_COLORS.get(self.multiplier, DEFAULT_FALLBACK_COLOR)
        else:
            self.color = (int(self.color[0]), int(self.color[1]), int(self.color[2]))

    @property
    def contour_int(self) -> np.ndarray:
        """Returns integer contour array suitable for OpenCV drawing functions."""
        return np.array(self.points, dtype=np.int32).reshape((-1, 1, 2))

    @property
    def contour_float(self) -> np.ndarray:
        """Returns float32 contour array for OpenCV geometric calculations."""
        return np.array(self.points, dtype=np.float32).reshape((-1, 1, 2))

    def contains_point(self, x: float, y: float) -> bool:
        """
        Determines whether the given coordinate (x, y) is inside or on the boundary
        of this scoring zone using cv2.pointPolygonTest.
        """
        result = cv2.pointPolygonTest(self.contour_float, (float(x), float(y)), False)
        return result >= 0.0

    def distance_to_point(self, x: float, y: float) -> float:
        """
        Computes signed distance from point (x, y) to polygon boundary:
        positive if inside, negative if outside, zero if exactly on edge.
        """
        return float(cv2.pointPolygonTest(self.contour_float, (float(x), float(y)), True))

    @property
    def centroid(self) -> Tuple[float, float]:
        """Calculates polygon centroid using image moments, or vertex mean as fallback."""
        m = cv2.moments(self.contour_float)
        if abs(m["m00"]) > 1e-6:
            return (float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"]))
        pts = np.array(self.points, dtype=np.float32)
        mean = np.mean(pts, axis=0)
        return (float(mean[0]), float(mean[1]))

    @property
    def area(self) -> float:
        """Calculates the 2D polygon area in board pixel units."""
        return float(cv2.contourArea(self.contour_float))

    def to_dict(self) -> Dict[str, Any]:
        """Serializes zone to JSON-compatible dictionary."""
        return {
            "name": self.name,
            "multiplier": int(self.multiplier),
            "points": [[round(p[0], 2), round(p[1], 2)] for p in self.points],
            "color": list(self.color) if self.color else list(DEFAULT_FALLBACK_COLOR),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ScoringZone:
        """Constructs a ScoringZone instance from a dictionary."""
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict for ScoringZone, got {type(data).__name__}")
        name = str(data.get("name", "Zone"))
        if "multiplier" not in data:
            raise ValueError("Missing required key 'multiplier' in zone data")
        multiplier = int(data["multiplier"])
        if "points" not in data:
            raise ValueError("Missing required key 'points' in zone data")
        points = data["points"]
        raw_color = data.get("color")
        color = tuple(raw_color) if raw_color is not None and len(raw_color) == 3 else None
        return cls(name=name, multiplier=multiplier, points=points, color=color)


class ArucoCalibrator:
    """
    Detects 4 corner ArUco markers on the Tumblin' Dice board,
    computes perspective homography transform matrix, and warps camera frames.
    """

    def __init__(
        self,
        dictionary_id: int = cv2.aruco.DICT_4X4_50,
        marker_ids: Optional[Sequence[int]] = None,
        board_width: int = DEFAULT_BOARD_WIDTH,
        board_height: int = DEFAULT_BOARD_HEIGHT,
        marker_inset_x: float = 0.0,
        marker_inset_y: float = 0.0,
        custom_destination_points: Optional[Sequence[Sequence[float]]] = None,
    ):
        self.dictionary_id = dictionary_id
        self.marker_ids = list(marker_ids) if marker_ids is not None else list(DEFAULT_MARKER_IDS)
        self.board_width = int(board_width)
        self.board_height = int(board_height)
        self.marker_inset_x = float(marker_inset_x)
        self.marker_inset_y = float(marker_inset_y)
        self.custom_destination_points = (
            [list(p) for p in custom_destination_points]
            if custom_destination_points is not None
            else None
        )

        self.homography_matrix: Optional[np.ndarray] = None
        self.inv_homography_matrix: Optional[np.ndarray] = None

        # Initialize ArUco detector
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(self.dictionary_id)
        if hasattr(cv2.aruco, "ArucoDetector"):
            self._detector = cv2.aruco.ArucoDetector(self._aruco_dict)
        else:
            self._detector = None

    def get_destination_points(self) -> np.ndarray:
        """
        Returns the (4, 2) destination corners on the rectified board plane:
        [Top-Left, Top-Right, Bottom-Right, Bottom-Left].
        """
        if self.custom_destination_points is not None:
            return np.array(self.custom_destination_points, dtype=np.float32)

        mx = self.marker_inset_x
        my = self.marker_inset_y
        w = float(self.board_width)
        h = float(self.board_height)

        return np.float32([
            [mx, my],           # Top-Left
            [w - mx, my],       # Top-Right
            [w - mx, h - my],   # Bottom-Right
            [mx, h - my],       # Bottom-Left
        ])

    def detect_markers(
        self, frame: np.ndarray
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, Tuple[float, float]]]:
        """
        Detects ArUco markers in the frame.
        Returns:
            corners_by_id: {marker_id: np.ndarray of shape (4, 2)}
            centers_by_id: {marker_id: (center_x, center_y)}
        """
        if frame is None or frame.size == 0:
            return {}, {}

        # Convert to grayscale if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        if self._detector is not None:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self._aruco_dict)

        corners_by_id: Dict[int, np.ndarray] = {}
        centers_by_id: Dict[int, Tuple[float, float]] = {}

        if ids is not None and len(ids) > 0:
            ids_flat = np.asarray(ids).flatten()
            for i, mid_val in enumerate(ids_flat):
                mid = int(mid_val)
                pts = corners[i][0]  # shape (4, 2)
                corners_by_id[mid] = pts
                center = np.mean(pts, axis=0)
                centers_by_id[mid] = (float(center[0]), float(center[1]))

        return corners_by_id, centers_by_id

    def compute_homography(
        self, src_points: Sequence[Sequence[float]], dst_points: Optional[Sequence[Sequence[float]]] = None
    ) -> np.ndarray:
        """
        Computes the 3x3 homography matrix from source points to destination points.
        Stores homography_matrix and inv_homography_matrix.
        """
        src = np.asarray(src_points, dtype=np.float32)
        if src.shape != (4, 2):
            raise ValueError(f"Source points must have shape (4, 2), got {src.shape}")

        if dst_points is not None:
            dst = np.asarray(dst_points, dtype=np.float32)
            if dst.shape != (4, 2):
                raise ValueError(f"Destination points must have shape (4, 2), got {dst.shape}")
        else:
            dst = self.get_destination_points()

        h_matrix = cv2.getPerspectiveTransform(src, dst)
        if h_matrix is None or h_matrix.shape != (3, 3):
            raise ValueError("Failed to compute valid 3x3 perspective transform matrix.")

        self.homography_matrix = h_matrix
        try:
            self.inv_homography_matrix = np.linalg.inv(h_matrix)
        except np.linalg.LinAlgError as err:
            raise ValueError("Computed homography matrix is singular and cannot be inverted.") from err

        return self.homography_matrix

    def calibrate_from_frame(self, frame: np.ndarray) -> bool:
        """
        Attempts to detect the 4 fiducials in the frame and calculate homography.
        First attempts to use configured marker_ids ([TL, TR, BR, BL]).
        Falls back to geometric ordering if at least 4 markers are found.

        Returns True on success, False if fewer than 4 markers found.
        """
        _, centers_by_id = self.detect_markers(frame)

        # 1. Check if all configured marker IDs are detected
        if all(mid in centers_by_id for mid in self.marker_ids):
            src_pts = np.float32([centers_by_id[mid] for mid in self.marker_ids])
            self.compute_homography(src_pts)
            return True

        # 2. Fallback: if at least 4 markers are detected, sort centers geometrically
        if len(centers_by_id) >= 4:
            available_pts = list(centers_by_id.values())[:4]
            ordered_pts = order_points_clockwise(available_pts)
            self.compute_homography(ordered_pts)
            return True

        return False

    def calibrate_from_points(
        self,
        src_points: Sequence[Sequence[float]],
        dst_points: Optional[Sequence[Sequence[float]]] = None,
    ) -> bool:
        """Directly calibrates from 4 user-supplied corner coordinates."""
        try:
            self.compute_homography(src_points, dst_points)
            return True
        except Exception as e:
            logger.warning(f"Manual calibration failed: {e}")
            return False

    def warp_frame(self, frame: np.ndarray) -> np.ndarray:
        """Transforms raw camera frame into normalized rectangular board plane."""
        if self.homography_matrix is None:
            raise RuntimeError("Calibrator has not been calibrated yet. Call calibrate_from_frame() first.")
        return cv2.warpPerspective(
            frame, self.homography_matrix, (self.board_width, self.board_height)
        )

    def unwarp_frame(self, warped_frame: np.ndarray, camera_shape: Tuple[int, int]) -> np.ndarray:
        """
        Transforms rectified board view back into the camera perspective.
        camera_shape: (height, width) of camera frame.
        """
        if self.inv_homography_matrix is None:
            raise RuntimeError("Calibrator has not been calibrated yet.")
        return cv2.warpPerspective(
            warped_frame, self.inv_homography_matrix, (camera_shape[1], camera_shape[0])
        )

    def transform_camera_to_board(self, x: float, y: float) -> Tuple[float, float]:
        """Maps point (x, y) from camera image coordinates to rectified board coordinates."""
        if self.homography_matrix is None:
            raise RuntimeError("Calibrator has not been calibrated yet.")
        pt = np.float32([[[float(x), float(y)]]])
        dst = cv2.perspectiveTransform(pt, self.homography_matrix)
        return float(dst[0][0][0]), float(dst[0][0][1])

    def transform_board_to_camera(self, x: float, y: float) -> Tuple[float, float]:
        """Maps point (x, y) from rectified board coordinates back to camera coordinates."""
        if self.inv_homography_matrix is None:
            raise RuntimeError("Calibrator has not been calibrated yet.")
        pt = np.float32([[[float(x), float(y)]]])
        dst = cv2.perspectiveTransform(pt, self.inv_homography_matrix)
        return float(dst[0][0][0]), float(dst[0][0][1])

    def transform_points(
        self, points: Sequence[Sequence[float]], to_board: bool = True
    ) -> np.ndarray:
        """Vectorized transformation of an (N, 2) array of coordinates."""
        if self.homography_matrix is None or self.inv_homography_matrix is None:
            raise RuntimeError("Calibrator has not been calibrated yet.")
        matrix = self.homography_matrix if to_board else self.inv_homography_matrix
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, matrix)
        return transformed.reshape(-1, 2)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes calibration configuration and homography matrix to dictionary."""
        return {
            "dictionary_id": int(self.dictionary_id),
            "marker_ids": list(self.marker_ids),
            "board_width": int(self.board_width),
            "board_height": int(self.board_height),
            "marker_inset_x": float(self.marker_inset_x),
            "marker_inset_y": float(self.marker_inset_y),
            "custom_destination_points": self.custom_destination_points,
            "homography_matrix": (
                self.homography_matrix.tolist() if self.homography_matrix is not None else None
            ),
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Restores calibration configuration from dictionary."""
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict for calibrator data, got {type(data).__name__}")
        self.dictionary_id = int(data.get("dictionary_id", self.dictionary_id))
        self.marker_ids = list(data.get("marker_ids", self.marker_ids))
        self.board_width = int(data.get("board_width", self.board_width))
        self.board_height = int(data.get("board_height", self.board_height))
        self.marker_inset_x = float(data.get("marker_inset_x", self.marker_inset_x))
        self.marker_inset_y = float(data.get("marker_inset_y", self.marker_inset_y))
        self.custom_destination_points = data.get("custom_destination_points")

        h_list = data.get("homography_matrix")
        if h_list is not None:
            self.homography_matrix = np.array(h_list, dtype=np.float32)
            self.inv_homography_matrix = np.linalg.inv(self.homography_matrix)
        else:
            self.homography_matrix = None
            self.inv_homography_matrix = None


class ZoneManager:
    """
    Manages collection of ScoringZones, spatial queries, HUD overlay rendering,
    and JSON persistence.
    """

    def __init__(
        self, board_width: int = DEFAULT_BOARD_WIDTH, board_height: int = DEFAULT_BOARD_HEIGHT
    ):
        self.board_width = int(board_width)
        self.board_height = int(board_height)
        self.zones: List[ScoringZone] = []

    def add_zone(self, zone: ScoringZone) -> None:
        """Adds a ScoringZone to the manager."""
        if not isinstance(zone, ScoringZone):
            raise TypeError(f"Expected ScoringZone instance, got {type(zone).__name__}")
        self.zones.append(zone)

    def remove_zone(self, index: int) -> ScoringZone:
        """Removes and returns the zone at index."""
        if 0 <= index < len(self.zones):
            return self.zones.pop(index)
        raise IndexError(f"Zone index {index} out of range (total zones: {len(self.zones)})")

    def remove_zone_by_name(self, name: str) -> bool:
        """Removes the first zone matching name. Returns True if found and removed."""
        for i, zone in enumerate(self.zones):
            if zone.name == name:
                self.zones.pop(i)
                return True
        return False

    def clear_zones(self) -> None:
        """Removes all scoring zones."""
        self.zones.clear()

    def get_zone(self, index: int) -> ScoringZone:
        """Returns zone at index."""
        return self.zones[index]

    def find_zone(self, x: float, y: float) -> Optional[ScoringZone]:
        """Finds the first scoring zone containing the point (x, y)."""
        for zone in self.zones:
            if zone.contains_point(x, y):
                return zone
        return None

    def get_multiplier(self, x: float, y: float) -> int:
        """Returns zone multiplier (1..6) if inside a zone, or 0 if off-board / out of bounds."""
        zone = self.find_zone(x, y)
        return zone.multiplier if zone is not None else 0

    def render_overlay(
        self,
        image: np.ndarray,
        alpha: float = 0.35,
        show_labels: bool = True,
        selected_zone_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Renders semi-transparent colored polygons, outlines, and multiplier badges on image.
        """
        if image is None or image.size == 0:
            raise ValueError("Input image cannot be None or empty.")

        output = image.copy()
        if not self.zones:
            return output

        # Draw semi-transparent fills
        overlay = image.copy()
        for idx, zone in enumerate(self.zones):
            color = zone.color or DEFAULT_ZONE_COLORS.get(zone.multiplier, DEFAULT_FALLBACK_COLOR)
            cv2.fillPoly(overlay, [zone.contour_int], color)

        cv2.addWeighted(overlay, alpha, output, 1.0 - alpha, 0, output)

        # Draw solid borders and multiplier text badges
        for idx, zone in enumerate(self.zones):
            is_selected = (selected_zone_idx == idx)
            border_color = (255, 255, 255) if is_selected else (zone.color or DEFAULT_FALLBACK_COLOR)
            thickness = 3 if is_selected else 2
            cv2.polylines(output, [zone.contour_int], isClosed=True, color=border_color, thickness=thickness)

            if show_labels:
                cx, cy = zone.centroid
                label = f"{zone.multiplier}x"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_thickness = 2
                (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
                bx = int(cx - tw / 2)
                by = int(cy + th / 2)

                # Draw dark badge background pill
                pad_x, pad_y = 6, 6
                cv2.rectangle(
                    output,
                    (bx - pad_x, by - th - pad_y),
                    (bx + tw + pad_x, by + baseline + pad_y),
                    (20, 20, 20),
                    -1,
                )
                cv2.rectangle(
                    output,
                    (bx - pad_x, by - th - pad_y),
                    (bx + tw + pad_x, by + baseline + pad_y),
                    border_color,
                    1,
                )
                cv2.putText(output, label, (bx, by), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        return output

    def to_dict(self) -> Dict[str, Any]:
        """Serializes zone manager configuration to dictionary."""
        return {
            "board_dimensions": {
                "width": self.board_width,
                "height": self.board_height,
            },
            "zones": [zone.to_dict() for zone in self.zones],
        }

    def from_dict(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> None:
        """
        Loads zones from either a structured dictionary or a raw list of zone dicts.
        """
        self.clear_zones()
        if isinstance(data, list):
            for item in data:
                self.add_zone(ScoringZone.from_dict(item))
        elif isinstance(data, dict):
            if "board_dimensions" in data:
                dims = data["board_dimensions"]
                self.board_width = int(dims.get("width", self.board_width))
                self.board_height = int(dims.get("height", self.board_height))
            raw_zones = data.get("zones", [])
            for item in raw_zones:
                self.add_zone(ScoringZone.from_dict(item))
        else:
            raise TypeError(f"Invalid zone data format: expected dict or list, got {type(data).__name__}")

    def save_to_json(self, filepath: str = DEFAULT_CONFIG_PATH) -> None:
        """Saves current zone setup to JSON."""
        abs_path = os.path.abspath(filepath)
        dirname = os.path.dirname(abs_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def load_from_json(self, filepath: str = DEFAULT_CONFIG_PATH) -> None:
        """Loads zone setup from JSON file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.from_dict(data)


class InteractiveZoneEditor:
    """
    Interactive polygon zone editor for Tumblin' Dice boards.
    Supports:
    - 1x..6x multiplier selection (keys '1'..'6')
    - Mouse left-click to add polygon vertices
    - Live rubber-band preview line to cursor
    - Right-click / 'u' to undo last point
    - 'Enter' / 'Space' / 'n' to complete polygon and save zone
    - 'c' / 'Escape' to cancel current polygon in-progress
    - 'd' / 'Backspace' to delete last completed zone
    - 's' to save zones to JSON
    - 'l' to load zones from JSON
    - 'q' to quit editor
    - Completely headless-testable event handlers
    """

    def __init__(
        self,
        zone_manager: Optional[ZoneManager] = None,
        calibrator: Optional[ArucoCalibrator] = None,
        config_path: str = DEFAULT_CONFIG_PATH,
        board_width: int = DEFAULT_BOARD_WIDTH,
        board_height: int = DEFAULT_BOARD_HEIGHT,
    ):
        self.zone_manager = zone_manager if zone_manager is not None else ZoneManager(board_width, board_height)
        self.calibrator = calibrator
        self.config_path = config_path

        self.active_multiplier: int = 1
        self.current_points: List[Tuple[int, int]] = []
        self.cursor_pos: Optional[Tuple[int, int]] = None
        self.status_message: str = "Ready. Click vertices to draw zone. Keys [1-6] select multiplier."
        self.is_running: bool = False

    def set_multiplier(self, multiplier: int) -> None:
        """Selects active multiplier (1..6)."""
        if multiplier not in VALID_MULTIPLIERS:
            raise ValueError(f"Multiplier must be in {VALID_MULTIPLIERS}, got {multiplier}")
        self.active_multiplier = multiplier
        self.status_message = f"Selected {self.active_multiplier}x multiplier zone."

    def add_point(self, x: int, y: int) -> None:
        """Adds a vertex (x, y) to the current polygon in progress."""
        pt = (int(x), int(y))
        self.current_points.append(pt)
        self.status_message = (
            f"Point {len(self.current_points)} added at ({x}, {y}). "
            f"Press Enter/N to finish ({self.active_multiplier}x)."
        )

    def undo_last_point(self) -> bool:
        """Removes the most recent vertex from the active polygon."""
        if self.current_points:
            removed = self.current_points.pop()
            self.status_message = f"Removed vertex {removed}. {len(self.current_points)} remaining."
            return True
        self.status_message = "No active vertices to undo."
        return False

    def finish_zone(self, name: Optional[str] = None) -> Optional[ScoringZone]:
        """
        Completes the current polygon, creates a ScoringZone with the active multiplier,
        and adds it to the ZoneManager. Automatically cycles to the next multiplier.
        """
        if len(self.current_points) < 3:
            self.status_message = (
                f"Cannot finish: polygon requires >= 3 points, currently have {len(self.current_points)}."
            )
            return None

        zone_name = name or f"Zone {self.active_multiplier}x"
        color = DEFAULT_ZONE_COLORS.get(self.active_multiplier, DEFAULT_FALLBACK_COLOR)
        zone = ScoringZone(
            name=zone_name,
            multiplier=self.active_multiplier,
            points=list(self.current_points),
            color=color,
        )
        self.zone_manager.add_zone(zone)
        pts_count = len(self.current_points)
        self.current_points.clear()

        # Cycle to next multiplier (1..6)
        next_mult = (self.active_multiplier % 6) + 1
        self.status_message = f"Saved '{zone.name}' ({zone.multiplier}x, {pts_count} pts). Next: {next_mult}x."
        self.active_multiplier = next_mult
        return zone

    def cancel_current_zone(self) -> None:
        """Cancels and discards the polygon currently in progress."""
        self.current_points.clear()
        self.status_message = "Cancelled current polygon."

    def delete_last_zone(self) -> Optional[ScoringZone]:
        """Deletes the most recently added zone from the manager."""
        if self.zone_manager.zones:
            removed = self.zone_manager.zones.pop()
            self.status_message = f"Deleted {removed.name} ({removed.multiplier}x)."
            return removed
        self.status_message = "No zones to delete."
        return None

    def save(self, filepath: Optional[str] = None) -> str:
        """Saves zones to JSON file."""
        target = filepath or self.config_path
        self.zone_manager.save_to_json(target)
        self.status_message = f"Saved {len(self.zone_manager.zones)} zones to '{target}'."
        return target

    def load(self, filepath: Optional[str] = None) -> str:
        """Loads zones from JSON file."""
        target = filepath or self.config_path
        self.zone_manager.load_from_json(target)
        self.status_message = f"Loaded {len(self.zone_manager.zones)} zones from '{target}'."
        return target

    def handle_mouse(self, event: int, x: int, y: int, flags: int, param: Any = None) -> None:
        """OpenCV mouse callback handler."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.add_point(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.undo_last_point()
        elif event == cv2.EVENT_MOUSEMOVE:
            self.cursor_pos = (x, y)

    def handle_key(self, key: Union[int, str]) -> bool:
        """
        Handles keyboard input events. Returns True to continue, False to exit.
        """
        if isinstance(key, int):
            if key < 0:
                return True
            code = key & 0xFF
            char = chr(code).lower() if 0 <= code < 128 else ""
        else:
            code = -1
            char = key.lower()

        # Multipliers '1'..'6'
        if char in "123456":
            self.set_multiplier(int(char))
            return True

        # Finish zone: Enter (\r or \n), Space (' '), or 'n'
        if char in ("\r", "\n", " ", "n") or code in (10, 13, 32):
            self.finish_zone()
            return True

        # Undo: 'u' or Backspace
        if char == "u" or code == 8:
            self.undo_last_point()
            return True

        # Cancel current: 'c' or Escape
        if char == "c" or code == 27:
            self.cancel_current_zone()
            return True

        # Delete last completed zone: 'd'
        if char == "d":
            self.delete_last_zone()
            return True

        # Save: 's'
        if char == "s":
            try:
                self.save()
            except Exception as e:
                self.status_message = f"Save failed: {e}"
            return True

        # Load: 'l'
        if char == "l":
            try:
                self.load()
            except Exception as e:
                self.status_message = f"Load failed: {e}"
            return True

        # Quit: 'q'
        if char == "q":
            self.is_running = False
            return False

        return True

    def render_ui(self, base_frame: np.ndarray) -> np.ndarray:
        """
        Renders HUD overlays:
        - Completed zones from ZoneManager
        - Current polygon in progress (lines, vertices, rubber-band to mouse)
        - Top status and control banner
        """
        if base_frame is None or base_frame.size == 0:
            raise ValueError("Base frame cannot be None or empty.")

        display = self.zone_manager.render_overlay(base_frame, alpha=0.35, show_labels=True)
        active_color = DEFAULT_ZONE_COLORS.get(self.active_multiplier, (255, 255, 255))

        # Draw in-progress polygon vertices and edges
        pts = self.current_points
        if pts:
            # Draw lines between existing vertices
            for i in range(len(pts) - 1):
                cv2.line(display, pts[i], pts[i + 1], active_color, 2, cv2.LINE_AA)

            # Draw rubber-band line from last point to current mouse cursor
            if self.cursor_pos is not None:
                cv2.line(display, pts[-1], self.cursor_pos, (200, 200, 200), 1, cv2.LINE_AA)

            # Draw vertex circles with indices
            for i, p in enumerate(pts):
                cv2.circle(display, p, 5, active_color, -1)
                cv2.circle(display, p, 7, (255, 255, 255), 1)
                cv2.putText(
                    display,
                    str(i + 1),
                    (p[0] + 8, p[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # Draw top HUD banner (header)
        h, w = display.shape[:2]
        banner_h = 70
        banner_overlay = display.copy()
        cv2.rectangle(banner_overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
        cv2.addWeighted(banner_overlay, 0.85, display, 0.15, 0, display)
        cv2.line(display, (0, banner_h), (w, banner_h), (80, 80, 80), 1)

        # Line 1: Multiplier selector & stats
        mult_str = "  ".join(
            [f"[{m}x*]" if m == self.active_multiplier else f"[{m}x]" for m in VALID_MULTIPLIERS]
        )
        line1 = f"Active: {self.active_multiplier}x  |  {mult_str}  |  Pts: {len(pts)}  |  Zones: {len(self.zone_manager.zones)}"
        cv2.putText(display, line1, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Line 2: Status / Instruction message
        cv2.putText(display, self.status_message, (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 255), 1, cv2.LINE_AA)

        # Line 3: Hotkey cheat-sheet
        line3 = "[1-6]: Mult | Click: Add Pt | R-Click/U: Undo | Enter/N: Finish | C: Cancel | D: Del | S: Save | L: Load | Q: Quit"
        cv2.putText(display, line3, (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

        return display

    def run(
        self,
        image_or_capture: Optional[Union[np.ndarray, cv2.VideoCapture]] = None,
        window_name: str = "Tumblin' Dice Zone Editor",
        max_iterations: Optional[int] = None,
    ) -> None:
        """
        Runs the interactive UI loop with OpenCV window.
        Gracefully handles headless environments where window creation fails.
        """
        self.is_running = True
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(window_name, self.handle_mouse)
        except cv2.error as e:
            logger.warning(f"Could not initialize OpenCV GUI window (headless environment?): {e}")
            self.is_running = False
            return

        is_video = isinstance(image_or_capture, cv2.VideoCapture)
        static_frame: Optional[np.ndarray] = None
        if isinstance(image_or_capture, np.ndarray):
            static_frame = image_or_capture
        elif not is_video:
            # Default blank canvas matching board dimensions
            static_frame = np.full(
                (self.zone_manager.board_height, self.zone_manager.board_width, 3), 240, dtype=np.uint8
            )

        iteration = 0
        while self.is_running:
            if max_iterations is not None and iteration >= max_iterations:
                break
            iteration += 1

            if is_video:
                ret, frame = image_or_capture.read()
                if not ret or frame is None:
                    break
                if self.calibrator is not None and self.calibrator.homography_matrix is not None:
                    frame = self.calibrator.warp_frame(frame)
            else:
                frame = static_frame.copy()

            rendered = self.render_ui(frame)
            try:
                cv2.imshow(window_name, rendered)
                k = cv2.waitKey(20)
                if not self.handle_key(k):
                    break
            except cv2.error:
                break

        try:
            cv2.destroyWindow(window_name)
        except Exception:
            pass
        self.is_running = False


class CalibrationPipeline:
    """
    High-level calibration pipeline linking:
    - ArUco 4-corner homography perspective warp
    - ScoringZone / ZoneManager point containment
    - Bidirectional coordinate queries (camera space <-> board space)
    - Full pipeline JSON persistence
    """

    def __init__(
        self,
        calibrator: Optional[ArucoCalibrator] = None,
        zone_manager: Optional[ZoneManager] = None,
        config_path: str = DEFAULT_CONFIG_PATH,
    ):
        self.calibrator = calibrator if calibrator is not None else ArucoCalibrator()
        self.zone_manager = (
            zone_manager
            if zone_manager is not None
            else ZoneManager(self.calibrator.board_width, self.calibrator.board_height)
        )
        self.config_path = config_path

    def calibrate_camera_frame(self, frame: np.ndarray) -> bool:
        """Calibrates homography warp from an overhead camera frame."""
        return self.calibrator.calibrate_from_frame(frame)

    def score_camera_point(
        self, camera_x: float, camera_y: float
    ) -> Tuple[float, float, int, Optional[str]]:
        """
        Maps a die center from camera coordinates into board coordinates,
        evaluates which scoring zone contains it, and returns:
        (board_x, board_y, multiplier, zone_name)
        """
        bx, by = self.calibrator.transform_camera_to_board(camera_x, camera_y)
        zone = self.zone_manager.find_zone(bx, by)
        multiplier = zone.multiplier if zone is not None else 0
        name = zone.name if zone is not None else None
        return bx, by, multiplier, name

    def score_board_point(self, board_x: float, board_y: float) -> Tuple[int, Optional[str]]:
        """Scores a point already in rectified board coordinates."""
        zone = self.zone_manager.find_zone(board_x, board_y)
        multiplier = zone.multiplier if zone is not None else 0
        name = zone.name if zone is not None else None
        return multiplier, name

    def process_frame(self, camera_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Processes a camera frame:
        1. Warps perspective into rectified board view and draws zones overlay.
        2. Projects zones back into camera view using inverse homography and draws overlay on camera frame.
        Returns:
            (rectified_view_with_zones, camera_view_with_zones)
        """
        if self.calibrator.homography_matrix is None:
            raise RuntimeError("Pipeline calibrator has not been calibrated.")

        warped = self.calibrator.warp_frame(camera_frame)
        warped_display = self.zone_manager.render_overlay(warped, alpha=0.35, show_labels=True)

        # Project zones back into camera frame
        cam_display = camera_frame.copy()
        if self.zone_manager.zones:
            cam_overlay = camera_frame.copy()
            for zone in self.zone_manager.zones:
                cam_pts = self.calibrator.transform_points(zone.points, to_board=False)
                cam_pts_int = np.array(cam_pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(cam_overlay, [cam_pts_int], zone.color)
            cv2.addWeighted(cam_overlay, 0.35, cam_display, 0.65, 0, cam_display)

            for zone in self.zone_manager.zones:
                cam_pts = self.calibrator.transform_points(zone.points, to_board=False)
                cam_pts_int = np.array(cam_pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(cam_display, [cam_pts_int], True, zone.color, 2)
                cx, cy = np.mean(cam_pts, axis=0)
                label = f"{zone.multiplier}x"
                cv2.putText(
                    cam_display,
                    label,
                    (int(cx - 10), int(cy + 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        return warped_display, cam_display

    def to_dict(self) -> Dict[str, Any]:
        """Serializes entire pipeline state (calibrator + zones)."""
        return {
            "version": "1.0",
            "board_dimensions": {
                "width": self.calibrator.board_width,
                "height": self.calibrator.board_height,
            },
            "calibrator": self.calibrator.to_dict(),
            "zones": [zone.to_dict() for zone in self.zone_manager.zones],
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Restores pipeline state from dictionary."""
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict, got {type(data).__name__}")
        if "calibrator" in data:
            self.calibrator.from_dict(data["calibrator"])
        if "board_dimensions" in data:
            dims = data["board_dimensions"]
            self.calibrator.board_width = int(dims.get("width", self.calibrator.board_width))
            self.calibrator.board_height = int(dims.get("height", self.calibrator.board_height))
            self.zone_manager.board_width = self.calibrator.board_width
            self.zone_manager.board_height = self.calibrator.board_height
        if "zones" in data:
            self.zone_manager.from_dict(data)

    def save(self, filepath: Optional[str] = None) -> str:
        """Saves pipeline configuration to JSON."""
        target = filepath or self.config_path
        abs_path = os.path.abspath(target)
        dirname = os.path.dirname(abs_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return abs_path

    def load(self, filepath: Optional[str] = None) -> str:
        """Loads pipeline configuration from JSON."""
        target = filepath or self.config_path
        if not os.path.exists(target):
            raise FileNotFoundError(f"Configuration file not found: {target}")
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.from_dict(data)
        return target


def generate_aruco_calibration_sheet(
    width: int = DEFAULT_BOARD_WIDTH,
    height: int = DEFAULT_BOARD_HEIGHT,
    marker_size: int = 100,
    margin: int = 50,
    marker_ids: Optional[Sequence[int]] = None,
    dictionary_id: int = cv2.aruco.DICT_4X4_50,
    draw_grid: bool = True,
) -> np.ndarray:
    """
    Generates a synthetic or printable ArUco calibration sheet with 4 corner fiducials:
    Top-Left, Top-Right, Bottom-Right, Bottom-Left.
    """
    sheet = np.full((height, width, 3), 255, dtype=np.uint8)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
    ids = list(marker_ids) if marker_ids is not None else list(DEFAULT_MARKER_IDS)

    # Positions for 4 corners (top-left x, top-left y)
    positions = [
        (margin, margin),                                          # TL
        (width - margin - marker_size, margin),                     # TR
        (width - margin - marker_size, height - margin - marker_size), # BR
        (margin, height - margin - marker_size),                   # BL
    ]

    for mid, (x, y) in zip(ids, positions):
        marker = cv2.aruco.generateImageMarker(aruco_dict, mid, marker_size)
        marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        sheet[y : y + marker_size, x : x + marker_size] = marker_bgr

        # Label under marker
        cv2.putText(
            sheet,
            f"ID {mid}",
            (x + 10, y + marker_size + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    if draw_grid:
        # Draw light guide lines around board boundaries
        cv2.rectangle(sheet, (margin, margin), (width - margin, height - margin), (220, 220, 220), 1)

    return sheet


def main() -> None:
    """Command-line interface for calibration and zone editor."""
    import argparse

    parser = argparse.ArgumentParser(description="Tumblin' Dice Calibration & Zone Editor")
    parser.add_argument("--camera", type=int, default=None, help="Camera device index (e.g. 0)")
    parser.add_argument("--image", type=str, default=None, help="Path to static image file")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to zones JSON config")
    parser.add_argument("--width", type=int, default=DEFAULT_BOARD_WIDTH, help="Warped board width in pixels")
    parser.add_argument("--height", type=int, default=DEFAULT_BOARD_HEIGHT, help="Warped board height in pixels")
    parser.add_argument("--generate-sheet", type=str, default=None, help="Generate ArUco calibration sheet image")
    parser.add_argument("--headless-check", action="store_true", help="Perform headless self-test")

    args = parser.parse_args()

    if args.generate_sheet:
        sheet = generate_aruco_calibration_sheet(args.width, args.height)
        cv2.imwrite(args.generate_sheet, sheet)
        print(f"Generated ArUco calibration sheet: {args.generate_sheet}")
        return

    if args.headless_check:
        print("Running calibration self-check...")
        sheet = generate_aruco_calibration_sheet(args.width, args.height)
        calibrator = ArucoCalibrator(board_width=args.width, board_height=args.height)
        success = calibrator.calibrate_from_frame(sheet)
        print(f"Calibration on synthetic sheet: {'SUCCESS' if success else 'FAILED'}")
        zone_mgr = ZoneManager(args.width, args.height)
        zone_mgr.add_zone(
            ScoringZone("Zone 1x", 1, [(100, 100), (700, 100), (700, 300), (100, 300)])
        )
        print(f"Zone test point (400, 200) multiplier: {zone_mgr.get_multiplier(400, 200)}x")
        return

    # Interactive zone editor
    calibrator = ArucoCalibrator(board_width=args.width, board_height=args.height)
    zone_mgr = ZoneManager(board_width=args.width, board_height=args.height)
    if os.path.exists(args.config):
        try:
            zone_mgr.load_from_json(args.config)
            print(f"Loaded existing configuration from {args.config}")
        except Exception as e:
            print(f"Could not load config: {e}")

    editor = InteractiveZoneEditor(
        zone_manager=zone_mgr,
        calibrator=calibrator,
        config_path=args.config,
        board_width=args.width,
        board_height=args.height,
    )

    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"Error: Could not open camera {args.camera}")
            sys.exit(1)
        editor.run(cap)
        cap.release()
    elif args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"Error: Could not read image {args.image}")
            sys.exit(1)
        editor.run(img)
    else:
        # Default blank board
        editor.run(None)


if __name__ == "__main__":
    main()
