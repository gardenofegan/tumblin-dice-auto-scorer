"""
detection.py - Tumblin' Dice Computer Vision Detection & Scoring Module

This module provides:
1. PlayerProfile and PlayerProfileManager:
   - Configures, serializes, and matches player dice colors in HSV color space.
   - Supports up to 10 distinct player colors with circular Hue statistics.
   - Accounts for achromatic colors (white, black) vs chromatic colors.
2. HSVColorSampler:
   - Samples base die material color while excluding contrasting pips.
   - Calculates circular mean/median Hue and standard deviation.
   - Generates calibrated lower/upper HSV boundaries for color masking.
3. D6PipCounter:
   - High-accuracy D6 pip counter (1 to 6) with both bright-on-dark and dark-on-bright support.
   - Circularity, area, and contour geometry validation.
   - Geometric confidence scoring for canonical D6 dot patterns.
4. DiceDetector:
   - Identifies candidate dice on board images via edge and contrast contours.
   - Applies Non-Maximum Suppression (NMS) to eliminate duplicate detections.
   - Integrates pip counting, base color sampling, and player assignment.
5. ZoneScorer:
   - Maps die coordinates to irregular polygon scoring zones via cv2.pointPolygonTest.
   - Computes live score: (Pip Count * Zone Multiplier).
   - Treats out-of-bounds/gutter dice as 0 points.
   - Compiles per-player round scores, active dice counts, and ranked leaderboards.
6. DetectionPipeline:
   - High-level coordinator processing camera/board frames into scoring snapshots.
   - Renders live HUD broadcast overlays with score badges, pips, and leaderboard.
   - Supports manual referee overrides (pip edit, player reassignment).
7. Synthetic Generators:
   - Generates realistic synthetic dice and multi-zone board images for testing.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

# Import calibration module if available in current workspace
try:
    from calibration import (
        DEFAULT_BOARD_HEIGHT,
        DEFAULT_BOARD_WIDTH,
        DEFAULT_ZONE_COLORS,
        VALID_MULTIPLIERS,
        ScoringZone,
        ZoneManager,
    )
except ImportError:
    DEFAULT_BOARD_WIDTH = 800
    DEFAULT_BOARD_HEIGHT = 1200
    VALID_MULTIPLIERS = (1, 2, 3, 4, 5, 6)
    DEFAULT_ZONE_COLORS = {
        1: (255, 140, 0),
        2: (50, 205, 50),
        3: (30, 30, 220),
        4: (0, 165, 255),
        5: (211, 0, 148),
        6: (0, 215, 255),
    }

    @dataclass
    class ScoringZone:  # type: ignore[no-redef]
        name: str
        multiplier: int
        points: List[Tuple[float, float]]
        color: Optional[Tuple[int, int, int]] = None

        def __post_init__(self) -> None:
            self.points = [(float(p[0]), float(p[1])) for p in self.points]
            if self.color is None:
                self.color = DEFAULT_ZONE_COLORS.get(self.multiplier, (180, 180, 180))

        @property
        def contour_float(self) -> np.ndarray:
            return np.array(self.points, dtype=np.float32).reshape((-1, 1, 2))

        def contains_point(self, x: float, y: float) -> bool:
            return cv2.pointPolygonTest(self.contour_float, (float(x), float(y)), False) >= 0.0

    class ZoneManager:  # type: ignore[no-redef]
        def __init__(self, board_width: int = DEFAULT_BOARD_WIDTH, board_height: int = DEFAULT_BOARD_HEIGHT):
            self.board_width = board_width
            self.board_height = board_height
            self.zones: List[ScoringZone] = []

        def add_zone(self, zone: ScoringZone) -> None:
            self.zones.append(zone)

        def find_zone(self, x: float, y: float) -> Optional[ScoringZone]:
            for zone in self.zones:
                if zone.contains_point(x, y):
                    return zone
            return None

        def get_multiplier(self, x: float, y: float) -> int:
            z = self.find_zone(x, y)
            return z.multiplier if z is not None else 0

logger = logging.getLogger(__name__)

# Standard preset color profiles for up to 10 players
DEFAULT_PLAYER_ROSTER: List[Dict[str, Any]] = [
    {"player_id": 1, "name": "Player 1", "color_name": "Red", "hsv_mean": (0.0, 240.0, 210.0), "display_bgr": (30, 30, 220)},
    {"player_id": 2, "name": "Player 2", "color_name": "Blue", "hsv_mean": (115.0, 240.0, 210.0), "display_bgr": (220, 100, 30)},
    {"player_id": 3, "name": "Player 3", "color_name": "Green", "hsv_mean": (60.0, 240.0, 200.0), "display_bgr": (50, 205, 50)},
    {"player_id": 4, "name": "Player 4", "color_name": "Yellow", "hsv_mean": (30.0, 240.0, 220.0), "display_bgr": (0, 215, 255)},
    {"player_id": 5, "name": "Player 5", "color_name": "Purple", "hsv_mean": (145.0, 220.0, 180.0), "display_bgr": (180, 0, 180)},
    {"player_id": 6, "name": "Player 6", "color_name": "Orange", "hsv_mean": (15.0, 240.0, 220.0), "display_bgr": (0, 140, 255)},
    {"player_id": 7, "name": "Player 7", "color_name": "Cyan", "hsv_mean": (90.0, 240.0, 220.0), "display_bgr": (255, 220, 0)},
    {"player_id": 8, "name": "Player 8", "color_name": "Magenta", "hsv_mean": (165.0, 220.0, 220.0), "display_bgr": (200, 50, 255)},
    {"player_id": 9, "name": "Player 9", "color_name": "White", "hsv_mean": (0.0, 10.0, 245.0), "display_bgr": (240, 240, 240)},
    {"player_id": 10, "name": "Player 10", "color_name": "Black", "hsv_mean": (0.0, 10.0, 30.0), "display_bgr": (40, 40, 40)},
]


# =============================================================================
# 1. Circular Color Statistics & HSV Utilities
# =============================================================================

def circular_hue_mean(hues: Union[Sequence[float], np.ndarray]) -> float:
    """
    Computes the circular mean for OpenCV Hues [0, 179] corresponding to [0, 360) degrees.
    Properly handles red hues wrapping across 0 and 179.
    """
    h_arr = np.asarray(hues, dtype=np.float64)
    if h_arr.size == 0:
        return 0.0
    # Map [0, 179] -> [0, 2*pi)
    angles = h_arr * (2.0 * np.pi / 180.0)
    sin_mean = np.mean(np.sin(angles))
    cos_mean = np.mean(np.cos(angles))
    mean_angle = np.arctan2(sin_mean, cos_mean) % (2.0 * np.pi)
    return float(mean_angle * (180.0 / (2.0 * np.pi)))


def circular_hue_distance(h1: float, h2: float) -> float:
    """
    Returns the shortest angular distance between two OpenCV hues in [0, 179].
    Result is in range [0, 90].
    """
    diff = abs(float(h1) - float(h2)) % 180.0
    return min(diff, 180.0 - diff)


def compute_hsv_distance(
    c1: Tuple[float, float, float],
    c2: Tuple[float, float, float],
    hue_weight: float = 2.5,
    sat_weight: float = 1.0,
    val_weight: float = 0.8,
) -> float:
    """
    Calculates perceptual distance between two HSV colors.
    Handles circular hue and gracefully distinguishes achromatic colors (white, black).
    """
    h1, s1, v1 = c1
    h2, s2, v2 = c2

    dh = circular_hue_distance(h1, h2) / 90.0  # Normalized [0, 1]
    ds = abs(float(s1) - float(s2)) / 255.0    # Normalized [0, 1]
    dv = abs(float(v1) - float(v2)) / 255.0    # Normalized [0, 1]

    # Achromatic check: low saturation or very low value
    is_achromatic1 = (s1 < 35.0 or v1 < 35.0)
    is_achromatic2 = (s2 < 35.0 or v2 < 35.0)

    if is_achromatic1 and is_achromatic2:
        # Both white/black/gray: Value and Saturation dominate, Hue is irrelevant
        return float(np.sqrt((1.2 * ds) ** 2 + (2.0 * dv) ** 2))
    elif is_achromatic1 != is_achromatic2:
        # One is chromatic, one is achromatic: heavily penalize mismatch
        return float(np.sqrt(1.6 + (1.5 * ds) ** 2 + (1.5 * dv) ** 2))
    else:
        # Both chromatic: Hue is the primary discriminator
        return float(np.sqrt((hue_weight * dh) ** 2 + (sat_weight * ds) ** 2 + (val_weight * dv) ** 2))


# =============================================================================
# 2. Player Profile & Profile Manager
# =============================================================================

@dataclass
class PlayerProfile:
    """
    Represents a registered player and their dice color profile in HSV space.
    """
    player_id: int
    name: str
    color_name: str
    hsv_mean: Tuple[float, float, float]
    hsv_std: Tuple[float, float, float] = (5.0, 20.0, 20.0)
    hsv_lower: Optional[Tuple[int, int, int]] = None
    hsv_upper: Optional[Tuple[int, int, int]] = None
    display_bgr: Tuple[int, int, int] = (255, 255, 255)

    def __post_init__(self) -> None:
        self.player_id = int(self.player_id)
        self.hsv_mean = (float(self.hsv_mean[0]), float(self.hsv_mean[1]), float(self.hsv_mean[2]))
        self.hsv_std = (float(self.hsv_std[0]), float(self.hsv_std[1]), float(self.hsv_std[2]))
        self.display_bgr = (int(self.display_bgr[0]), int(self.display_bgr[1]), int(self.display_bgr[2]))

        if self.hsv_lower is None or self.hsv_upper is None:
            self._compute_default_bounds()

    def _compute_default_bounds(self, delta_h: int = 15, delta_s: int = 50, delta_v: int = 60) -> None:
        h, s, v = self.hsv_mean
        lower_h = int(max(0, h - delta_h))
        upper_h = int(min(179, h + delta_h))
        lower_s = int(max(0, s - delta_s))
        upper_s = int(min(255, s + delta_s))
        lower_v = int(max(0, v - delta_v))
        upper_v = int(min(255, v + delta_v))
        self.hsv_lower = (lower_h, lower_s, lower_v)
        self.hsv_upper = (upper_h, upper_s, upper_v)

    def distance_to(self, hsv: Tuple[float, float, float]) -> float:
        """Computes distance from a target HSV color to this profile's mean color."""
        return compute_hsv_distance(self.hsv_mean, hsv)

    def matches(self, hsv: Tuple[float, float, float], max_distance: float = 0.8) -> bool:
        """Returns True if the given HSV color is within max_distance of this profile."""
        return self.distance_to(hsv) <= max_distance

    def to_dict(self) -> Dict[str, Any]:
        """Serializes player profile to dictionary."""
        return {
            "player_id": self.player_id,
            "name": self.name,
            "color_name": self.color_name,
            "hsv_mean": [round(c, 2) for c in self.hsv_mean],
            "hsv_std": [round(c, 2) for c in self.hsv_std],
            "hsv_lower": list(self.hsv_lower) if self.hsv_lower else None,
            "hsv_upper": list(self.hsv_upper) if self.hsv_upper else None,
            "display_bgr": list(self.display_bgr),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlayerProfile:
        """Constructs PlayerProfile from dictionary."""
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict for PlayerProfile, got {type(data).__name__}")
        player_id = int(data["player_id"])
        name = str(data.get("name", f"Player {player_id}"))
        color_name = str(data.get("color_name", "Custom"))
        hsv_mean = tuple(data["hsv_mean"])
        hsv_std = tuple(data.get("hsv_std", (5.0, 20.0, 20.0)))
        raw_lower = data.get("hsv_lower")
        raw_upper = data.get("hsv_upper")
        hsv_lower = tuple(raw_lower) if raw_lower is not None else None
        hsv_upper = tuple(raw_upper) if raw_upper is not None else None
        display_bgr = tuple(data.get("display_bgr", (255, 255, 255)))
        return cls(
            player_id=player_id,
            name=name,
            color_name=color_name,
            hsv_mean=hsv_mean,  # type: ignore[arg-type]
            hsv_std=hsv_std,    # type: ignore[arg-type]
            hsv_lower=hsv_lower,  # type: ignore[arg-type]
            hsv_upper=hsv_upper,  # type: ignore[arg-type]
            display_bgr=display_bgr,  # type: ignore[arg-type]
        )


class PlayerProfileManager:
    """
    Manages the roster of player profiles, persistence to JSON, and color matching.
    Supports 2 to 10 players.
    """

    def __init__(self, profiles: Optional[Sequence[PlayerProfile]] = None):
        self.profiles: Dict[int, PlayerProfile] = {}
        if profiles:
            for p in profiles:
                self.add_profile(p)

    def add_profile(self, profile: PlayerProfile) -> None:
        """Adds or updates a player profile."""
        if not isinstance(profile, PlayerProfile):
            raise TypeError(f"Expected PlayerProfile, got {type(profile).__name__}")
        self.profiles[profile.player_id] = profile

    def remove_profile(self, player_id: int) -> bool:
        """Removes profile by player_id. Returns True if found and removed."""
        return self.profiles.pop(player_id, None) is not None

    def get_profile(self, player_id: int) -> Optional[PlayerProfile]:
        """Gets profile by player_id."""
        return self.profiles.get(player_id)

    def get_all_profiles(self) -> List[PlayerProfile]:
        """Returns all player profiles sorted by player_id."""
        return sorted(self.profiles.values(), key=lambda p: p.player_id)

    def match_color(
        self, hsv: Tuple[float, float, float], max_distance: float = 0.8
    ) -> Tuple[Optional[PlayerProfile], float]:
        """
        Finds the closest registered player profile for a given HSV color.
        Returns (PlayerProfile, distance) or (None, min_distance) if beyond max_distance.
        """
        if not self.profiles:
            return None, float("inf")

        best_profile: Optional[PlayerProfile] = None
        min_dist = float("inf")

        for profile in self.profiles.values():
            dist = profile.distance_to(hsv)
            if dist < min_dist:
                min_dist = dist
                best_profile = profile

        if min_dist <= max_distance:
            return best_profile, min_dist
        return None, min_dist

    def save_to_json(self, filepath: str) -> None:
        """Saves player profiles to JSON file."""
        data = {
            "num_players": len(self.profiles),
            "players": [p.to_dict() for p in self.get_all_profiles()],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_from_json(self, filepath: str) -> None:
        """Loads player profiles from JSON file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Player profile config file not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        players_data = data.get("players", data) if isinstance(data, dict) else data
        if not isinstance(players_data, list):
            raise ValueError(f"Invalid player profile JSON format in {filepath}")

        self.profiles.clear()
        for p_data in players_data:
            self.add_profile(PlayerProfile.from_dict(p_data))

    @classmethod
    def create_default_roster(cls, num_players: int = 4) -> PlayerProfileManager:
        """
        Creates a pre-configured roster with default colors for up to 10 players.
        """
        if num_players < 1 or num_players > 10:
            raise ValueError(f"Tumblin' Dice supports 1 to 10 players, requested {num_players}")

        manager = cls()
        for i in range(num_players):
            preset = DEFAULT_PLAYER_ROSTER[i]
            profile = PlayerProfile(
                player_id=preset["player_id"],
                name=preset["name"],
                color_name=preset["color_name"],
                hsv_mean=preset["hsv_mean"],
                display_bgr=preset["display_bgr"],
            )
            manager.add_profile(profile)
        return manager


# =============================================================================
# 3. HSV Color Sampler (with Pip Exclusion)
# =============================================================================

class HSVColorSampler:
    """
    Samples die base material color in HSV space while excluding high-contrast pips.
    Uses circular statistics for hue and contrast difference masking.
    """

    def __init__(self, pip_exclusion_threshold: float = 35.0, border_margin_pct: float = 0.10):
        self.pip_exclusion_threshold = float(pip_exclusion_threshold)
        self.border_margin_pct = float(border_margin_pct)

    def sample_color_from_roi(
        self,
        roi_bgr: np.ndarray,
        exclude_pips: bool = True,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, float]:
        """
        Samples base material HSV color from a cropped die face ROI.
        If exclude_pips is True, isolates and removes high-contrast pip pixels.
        Returns: (H, S, V) in OpenCV ranges [0..179, 0..255, 0..255].
        """
        if roi_bgr is None or roi_bgr.size == 0:
            raise ValueError("ROI image cannot be None or empty.")

        h, w = roi_bgr.shape[:2]
        total_pixels = h * w if roi_bgr.ndim >= 2 else roi_bgr.size
        if total_pixels < 20:
            raise ValueError(f"ROI too small for sampling ({total_pixels} pixels). Minimum is 20 pixels.")

        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        if mask is not None:
            valid_mask = (mask == 255)
        else:
            valid_mask = np.ones((h, w), dtype=bool)

        if not exclude_pips:
            selected_hsv = hsv[valid_mask]
            if selected_hsv.shape[0] == 0:
                selected_hsv = hsv.reshape(-1, 3)
            return (
                circular_hue_mean(selected_hsv[:, 0]),
                float(np.median(selected_hsv[:, 1])),
                float(np.median(selected_hsv[:, 2])),
            )

        # Pip exclusion using intensity contrast
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        masked_gray = gray[valid_mask]
        med_val = float(np.median(masked_gray)) if masked_gray.size > 0 else float(np.median(gray))

        # Pips have significant intensity deviation from the median die body
        contrast = np.abs(gray.astype(np.float32) - med_val)
        pip_mask = contrast > self.pip_exclusion_threshold

        body_mask = valid_mask & (~pip_mask)
        body_pixels_hsv = hsv[body_mask]

        # If too many pixels were masked (or image is monochromatic), use outer border ring
        if body_pixels_hsv.shape[0] < 15:
            my = max(1, int(h * self.border_margin_pct))
            mx = max(1, int(w * self.border_margin_pct))
            ring_mask = np.zeros((h, w), dtype=bool)
            ring_mask[:my, :] = True
            ring_mask[-my:, :] = True
            ring_mask[:, :mx] = True
            ring_mask[:, -mx:] = True
            combined_ring = ring_mask & valid_mask
            body_pixels_hsv = hsv[combined_ring] if np.sum(combined_ring) >= 10 else hsv[valid_mask]

        if body_pixels_hsv.shape[0] == 0:
            # Fallback to all pixels
            body_pixels_hsv = hsv.reshape(-1, 3)

        hues = body_pixels_hsv[:, 0]
        sats = body_pixels_hsv[:, 1]
        vals = body_pixels_hsv[:, 2]

        mean_h = circular_hue_mean(hues)
        med_s = float(np.median(sats))
        med_v = float(np.median(vals))

        return (mean_h, med_s, med_v)

    def sample_color_from_circle(
        self,
        frame_bgr: np.ndarray,
        center: Tuple[int, int],
        radius: int = 20,
        exclude_pips: bool = True,
    ) -> Tuple[float, float, float]:
        """
        Samples color from an on-screen circular sampling target.
        Used during pre-game wizard when placing a die under the crosshair.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Frame image cannot be None or empty.")

        cx, cy = int(center[0]), int(center[1])
        r = int(radius)
        fh, fw = frame_bgr.shape[:2]

        x1 = max(0, cx - r)
        y1 = max(0, cy - r)
        x2 = min(fw, cx + r)
        y2 = min(fh, cy + r)

        sub_img = frame_bgr[y1:y2, x1:x2]
        if sub_img.size == 0:
            raise ValueError(f"Sampling circle ({cx}, {cy}, r={r}) is outside frame bounds ({fw}x{fh}).")

        # Mask circle within sub_img
        sh, sw = sub_img.shape[:2]
        mask = np.zeros((sh, sw), dtype=np.uint8)
        cv2.circle(mask, (cx - x1, cy - y1), r, 255, -1)

        return self.sample_color_from_roi(sub_img, exclude_pips=exclude_pips, mask=mask)

    def create_player_profile(
        self,
        player_id: int,
        name: str,
        color_name: str,
        sample_img: np.ndarray,
        center: Optional[Tuple[int, int]] = None,
        radius: int = 20,
        display_bgr: Optional[Tuple[int, int, int]] = None,
    ) -> PlayerProfile:
        """
        Creates a calibrated PlayerProfile directly from a sampled die image.
        """
        if center is not None:
            hsv_mean = self.sample_color_from_circle(sample_img, center=center, radius=radius)
        else:
            hsv_mean = self.sample_color_from_roi(sample_img)

        # Estimate display BGR color from HSV
        if display_bgr is None:
            h_int = int(round(hsv_mean[0]))
            s_int = int(round(hsv_mean[1]))
            v_int = int(round(hsv_mean[2]))
            hsv_pix = np.uint8([[[h_int, s_int, v_int]]])
            bgr_pix = cv2.cvtColor(hsv_pix, cv2.COLOR_HSV2BGR)[0][0]
            display_bgr = (int(bgr_pix[0]), int(bgr_pix[1]), int(bgr_pix[2]))

        return PlayerProfile(
            player_id=player_id,
            name=name,
            color_name=color_name,
            hsv_mean=hsv_mean,
            display_bgr=display_bgr,
        )


# =============================================================================
# 4. D6 Pip Counter
# =============================================================================

@dataclass
class PipCountResult:
    """
    Result of D6 pip counting on a cropped die ROI.
    """
    pip_count: int                              # 1..6, or 0 if undetected
    confidence: float                           # 0.0..1.0
    pip_centers: List[Tuple[float, float]]      # Centers (x, y) relative to ROI
    polarity: str                               # 'bright' (light pips) or 'dark' (dark pips)
    geometric_score: float = 0.0                # Adherence to D6 pip geometry
    debug_image: Optional[np.ndarray] = None


class D6PipCounter:
    """
    Counts pips (1 to 6) on a cropped die face image.
    Supports both bright pips on dark dice and dark pips on light dice.
    Validates circularity, area proportions, and canonical D6 dot geometry.
    """

    def __init__(
        self,
        eval_size: int = 120,
        min_circularity: float = 0.42,
        min_area_ratio: float = 0.003,
        max_area_ratio: float = 0.12,
        border_exclusion_ratio: float = 0.07,
    ):
        self.eval_size = eval_size
        self.min_circularity = float(min_circularity)
        self.min_area_ratio = float(min_area_ratio)
        self.max_area_ratio = float(max_area_ratio)
        self.border_exclusion_ratio = float(border_exclusion_ratio)

    def count_pips(self, die_roi: np.ndarray) -> PipCountResult:
        """
        Analyzes a cropped die ROI and returns the detected pip count (1..6) and confidence.
        """
        if die_roi is None or die_roi.size == 0:
            return PipCountResult(pip_count=0, confidence=0.0, pip_centers=[], polarity="unknown")

        orig_h, orig_w = die_roi.shape[:2]
        if orig_h < 10 or orig_w < 10:
            return PipCountResult(pip_count=0, confidence=0.0, pip_centers=[], polarity="unknown")

        # Resize to standardized evaluation size for scale-invariant morphology
        eval_size = self.eval_size
        resized = cv2.resize(die_roi, (eval_size, eval_size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized

        total_pixels = float(eval_size * eval_size)
        min_area = self.min_area_ratio * total_pixels
        max_area = self.max_area_ratio * total_pixels
        border_margin = int(self.border_exclusion_ratio * eval_size)

        # Compute outer margin border mean to determine background polarity
        mask_border = np.zeros((eval_size, eval_size), dtype=bool)
        mask_border[:border_margin, :] = True
        mask_border[-border_margin:, :] = True
        mask_border[:, :border_margin] = True
        mask_border[:, -border_margin:] = True
        border_mean = float(np.mean(gray[mask_border]))

        # Test both pip polarities (bright pips and dark pips)
        candidate_results: List[Tuple[int, float, float, List[Tuple[float, float]], str, np.ndarray]] = []

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        for polarity in ("bright", "dark"):
            if polarity == "bright":
                # Pips are brighter than die body
                thresh_val = max(35.0, min(240.0, border_mean + 28.0))
                _, binary = cv2.threshold(gray, int(thresh_val), 255, cv2.THRESH_BINARY)
            else:
                # Pips are darker than die body
                thresh_val = max(15.0, min(220.0, border_mean - 28.0))
                _, binary = cv2.threshold(gray, int(thresh_val), 255, cv2.THRESH_BINARY_INV)

            # Morphological opening to detach minor border artifacts
            opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

            # Find contours
            contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            valid_pips: List[Tuple[float, float, float, float]] = []  # (cx, cy, area, circularity)
            for cnt in contours:
                area = float(cv2.contourArea(cnt))
                if area < min_area or area > max_area:
                    continue

                perim = float(cv2.arcLength(cnt, True))
                if perim <= 0:
                    continue

                circularity = (4.0 * np.pi * area) / (perim * perim)
                if circularity < self.min_circularity:
                    continue

                m = cv2.moments(cnt)
                if abs(m["m00"]) < 1e-5:
                    continue

                cx = float(m["m10"] / m["m00"])
                cy = float(m["m01"] / m["m00"])

                # Discard pips that touch or sit on the extreme die border
                if cx < border_margin or cx > (eval_size - border_margin):
                    continue
                if cy < border_margin or cy > (eval_size - border_margin):
                    continue

                valid_pips.append((cx, cy, area, circularity))

            count = len(valid_pips)
            if 1 <= count <= 6:
                # Map centers to normalized [0, 1] for geometry scoring
                norm_pts = [(p[0] / eval_size, p[1] / eval_size) for p in valid_pips]
                geom_score = self._score_geometry(norm_pts, count)
                avg_circ = float(np.mean([p[3] for p in valid_pips]))
                # Combined confidence
                conf = 0.5 * avg_circ + 0.5 * geom_score
                # Scale centers back to original ROI dimensions
                scale_x = orig_w / float(eval_size)
                scale_y = orig_h / float(eval_size)
                orig_centers = [(p[0] * scale_x, p[1] * scale_y) for p in valid_pips]
                candidate_results.append((count, conf, geom_score, orig_centers, polarity, opened))

        if not candidate_results:
            # Fallback attempt using Otsu thresholding if fixed threshold missed subtle pips
            candidate_results = self._fallback_otsu_counting(gray, orig_w, orig_h, border_mean)

        if not candidate_results:
            return PipCountResult(pip_count=0, confidence=0.0, pip_centers=[], polarity="unknown")

        # Pick candidate with highest confidence (preferring valid 1..6 count with best geometry)
        candidate_results.sort(key=lambda c: (1 <= c[0] <= 6, c[1]), reverse=True)
        best = candidate_results[0]

        return PipCountResult(
            pip_count=best[0],
            confidence=round(best[1], 3),
            pip_centers=best[3],
            polarity=best[4],
            geometric_score=round(best[2], 3),
            debug_image=best[5],
        )

    def _fallback_otsu_counting(
        self, gray: np.ndarray, orig_w: int, orig_h: int, border_mean: float
    ) -> List[Tuple[int, float, float, List[Tuple[float, float]], str, np.ndarray]]:
        """Fallback evaluation using Otsu thresholding."""
        eval_size = self.eval_size
        total_pixels = float(eval_size * eval_size)
        min_area = self.min_area_ratio * total_pixels
        max_area = self.max_area_ratio * total_pixels
        border_margin = int(self.border_exclusion_ratio * eval_size)

        results = []
        for polarity in ("bright", "dark"):
            flag = cv2.THRESH_BINARY if polarity == "bright" else cv2.THRESH_BINARY_INV
            _, binary = cv2.threshold(gray, 0, 255, flag + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            valid_pips = []
            for cnt in contours:
                area = float(cv2.contourArea(cnt))
                if area < min_area or area > max_area:
                    continue
                perim = float(cv2.arcLength(cnt, True))
                if perim <= 0:
                    continue
                circ = (4.0 * np.pi * area) / (perim * perim)
                if circ < self.min_circularity:
                    continue
                m = cv2.moments(cnt)
                if abs(m["m00"]) < 1e-5:
                    continue
                cx, cy = float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])
                if cx < border_margin or cx > (eval_size - border_margin):
                    continue
                if cy < border_margin or cy > (eval_size - border_margin):
                    continue
                valid_pips.append((cx, cy, area, circ))

            count = len(valid_pips)
            if 1 <= count <= 6:
                norm_pts = [(p[0] / eval_size, p[1] / eval_size) for p in valid_pips]
                geom_score = self._score_geometry(norm_pts, count)
                avg_circ = float(np.mean([p[3] for p in valid_pips]))
                conf = 0.5 * avg_circ + 0.5 * geom_score
                scale_x = orig_w / float(eval_size)
                scale_y = orig_h / float(eval_size)
                orig_centers = [(p[0] * scale_x, p[1] * scale_y) for p in valid_pips]
                results.append((count, conf, geom_score, orig_centers, polarity, opened))
        return results

    def _score_geometry(self, points_norm: Sequence[Tuple[float, float]], count: int) -> float:
        """
        Computes geometric plausibility score [0.0..1.0] for canonical D6 dot layouts.
        """
        pts = np.asarray(points_norm, dtype=np.float32)
        center = np.array([0.5, 0.5], dtype=np.float32)

        if count == 1:
            # Face 1: Single dot centered
            d = float(np.linalg.norm(pts[0] - center))
            return max(0.0, 1.0 - 3.0 * d)

        elif count == 2:
            # Face 2: Two diagonal dots symmetric across center
            mid = (pts[0] + pts[1]) / 2.0
            d_mid = float(np.linalg.norm(mid - center))
            dist = float(np.linalg.norm(pts[0] - pts[1]))
            # Distance between pips should be significant (> 0.25)
            spacing_score = 1.0 if dist > 0.25 else 0.4
            return max(0.0, 1.0 - 2.5 * d_mid) * spacing_score

        elif count == 3:
            # Face 3: One dot at center, two diagonal dots symmetric
            dists = [float(np.linalg.norm(p - center)) for p in pts]
            min_idx = int(np.argmin(dists))
            center_d = dists[min_idx]
            other_pts = [pts[i] for i in range(3) if i != min_idx]
            mid = (other_pts[0] + other_pts[1]) / 2.0
            d_mid = float(np.linalg.norm(mid - center))
            return max(0.0, 1.0 - 2.5 * center_d) * max(0.0, 1.0 - 2.5 * d_mid)

        elif count == 4:
            # Face 4: Four corner dots, none at center
            dists = [float(np.linalg.norm(p - center)) for p in pts]
            if min(dists) < 0.16:
                return 0.2  # Center dot should not exist on Face 4
            mean_pt = np.mean(pts, axis=0)
            return max(0.0, 1.0 - 2.5 * float(np.linalg.norm(mean_pt - center)))

        elif count == 5:
            # Face 5: Four corner dots + one center dot
            dists = [float(np.linalg.norm(p - center)) for p in pts]
            min_d = min(dists)
            mean_pt = np.mean(pts, axis=0)
            return max(0.0, 1.0 - 2.5 * min_d) * max(0.0, 1.0 - 2.5 * float(np.linalg.norm(mean_pt - center)))

        elif count == 6:
            # Face 6: Two rows/columns of 3 dots, none at center
            dists = [float(np.linalg.norm(p - center)) for p in pts]
            if min(dists) < 0.15:
                return 0.2  # Center dot should not exist on Face 6
            mean_pt = np.mean(pts, axis=0)
            return max(0.0, 1.0 - 2.5 * float(np.linalg.norm(mean_pt - center)))

        return 0.0


# =============================================================================
# 5. Detected Die Model & Candidate Bounding Box Utilities
# =============================================================================

@dataclass
class DetectedDie:
    """
    Represents a detected physical die on the game board.
    """
    die_id: int
    center: Tuple[float, float]                 # (x, y) on board
    bbox: Tuple[int, int, int, int]             # (x, y, w, h)
    pip_count: int                              # 1..6
    pip_confidence: float                       # 0.0..1.0
    pip_centers: List[Tuple[float, float]] = field(default_factory=list)
    player_id: Optional[int] = None
    player_name: Optional[str] = None
    player_color: Optional[str] = None
    color_distance: float = 0.0
    hsv_sampled: Optional[Tuple[float, float, float]] = None
    display_bgr: Tuple[int, int, int] = (255, 255, 255)
    zone_name: Optional[str] = None
    zone_multiplier: int = 0
    score: int = 0
    is_out_of_bounds: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serializes detected die to dictionary."""
        return {
            "die_id": self.die_id,
            "center": [round(self.center[0], 2), round(self.center[1], 2)],
            "bbox": list(self.bbox),
            "pip_count": self.pip_count,
            "pip_confidence": round(self.pip_confidence, 3),
            "player_id": self.player_id,
            "player_name": self.player_name,
            "player_color": self.player_color,
            "color_distance": round(self.color_distance, 3),
            "hsv_sampled": [round(c, 2) for c in self.hsv_sampled] if self.hsv_sampled else None,
            "zone_name": self.zone_name,
            "zone_multiplier": self.zone_multiplier,
            "score": self.score,
            "is_out_of_bounds": self.is_out_of_bounds,
        }


def compute_box_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Computes Intersection-over-Union (IoU) between two bounding boxes (x, y, w, h)."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter_area == 0:
        return 0.0
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    return float(inter_area / union_area) if union_area > 0 else 0.0


def non_maximum_suppression(
    boxes: Sequence[Tuple[int, int, int, int]], scores: Sequence[float], iou_threshold: float = 0.35
) -> List[int]:
    """
    Standard Non-Maximum Suppression (NMS) to eliminate overlapping redundant candidate boxes.
    Returns list of chosen indices.
    """
    if not boxes:
        return []

    indices = list(range(len(boxes)))
    # Sort descending by score
    indices.sort(key=lambda i: scores[i], reverse=True)

    keep: List[int] = []
    while indices:
        current = indices.pop(0)
        keep.append(current)
        indices = [
            i for i in indices if compute_box_iou(boxes[current], boxes[i]) < iou_threshold
        ]
    return keep


# =============================================================================
# 6. Point-in-Polygon Zone Scorer
# =============================================================================

@dataclass
class PlayerRoundScore:
    """Aggregated score and active dice for a specific player."""
    player_id: int
    player_name: str
    color_name: str
    display_bgr: Tuple[int, int, int]
    dice: List[DetectedDie] = field(default_factory=list)
    total_score: int = 0
    active_dice_count: int = 0
    out_of_bounds_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "player_id": self.player_id,
            "player_name": self.player_name,
            "color_name": self.color_name,
            "total_score": self.total_score,
            "active_dice_count": self.active_dice_count,
            "out_of_bounds_count": self.out_of_bounds_count,
            "dice": [d.to_dict() for d in self.dice],
        }


@dataclass
class DiceScoringResult:
    """Complete snapshot of scoring on the board."""
    dice: List[DetectedDie] = field(default_factory=list)
    player_scores: Dict[int, int] = field(default_factory=dict)
    player_round_scores: Dict[int, PlayerRoundScore] = field(default_factory=dict)
    leaderboard: List[Dict[str, Any]] = field(default_factory=list)
    out_of_bounds_dice: List[DetectedDie] = field(default_factory=list)
    unassigned_dice: List[DetectedDie] = field(default_factory=list)
    total_board_score: int = 0
    active_dice_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_board_score": self.total_board_score,
            "active_dice_count": self.active_dice_count,
            "player_scores": self.player_scores,
            "leaderboard": self.leaderboard,
            "dice": [d.to_dict() for d in self.dice],
            "out_of_bounds_count": len(self.out_of_bounds_dice),
            "unassigned_count": len(self.unassigned_dice),
        }


class ZoneScorer:
    """
    Evaluates detected dice positions against scoring zones.
    Computes (Pip Count * Zone Multiplier) and aggregates player leaderboards.
    """

    def __init__(
        self,
        zones_source: Optional[Union[ZoneManager, Sequence[ScoringZone], str, List[Dict[str, Any]]]] = None,
    ):
        self.zone_manager: Optional[ZoneManager] = None
        self.zones: List[ScoringZone] = []

        if zones_source is not None:
            self.set_zones(zones_source)

    def set_zones(
        self, zones_source: Union[ZoneManager, Sequence[ScoringZone], str, List[Dict[str, Any]]]
    ) -> None:
        """Configures scoring zones from ZoneManager, list of zones, JSON path, or dicts."""
        if isinstance(zones_source, ZoneManager):
            self.zone_manager = zones_source
            self.zones = list(zones_source.zones)
        elif isinstance(zones_source, str):
            # Load from JSON file path
            if not os.path.exists(zones_source):
                raise FileNotFoundError(f"Zone config file not found: {zones_source}")
            zm = ZoneManager()
            if hasattr(zm, "load_from_json"):
                zm.load_from_json(zones_source)
                self.zone_manager = zm
                self.zones = list(zm.zones)
            else:
                with open(zones_source, "r", encoding="utf-8") as f:
                    data = json.load(f)
                z_list = data.get("zones", data) if isinstance(data, dict) else data
                self.zones = [ScoringZone(z["name"], z["multiplier"], z["points"]) for z in z_list]
        elif isinstance(zones_source, Sequence):
            self.zones = []
            for item in zones_source:
                if isinstance(item, ScoringZone):
                    self.zones.append(item)
                elif isinstance(item, dict):
                    self.zones.append(
                        ScoringZone(item["name"], int(item["multiplier"]), item["points"])
                    )
                else:
                    raise TypeError(f"Unsupported zone item type: {type(item).__name__}")
        else:
            raise TypeError(f"Unsupported zones_source type: {type(zones_source).__name__}")

    def find_zone(self, x: float, y: float) -> Optional[ScoringZone]:
        """Finds scoring zone containing point (x, y) using cv2.pointPolygonTest."""
        if self.zone_manager is not None:
            return self.zone_manager.find_zone(x, y)
        for zone in self.zones:
            if zone.contains_point(x, y):
                return zone
        return None

    def get_multiplier(self, x: float, y: float) -> int:
        """Returns zone multiplier at (x, y), or 0 if out of bounds."""
        zone = self.find_zone(x, y)
        return zone.multiplier if zone is not None else 0

    def score_die(self, die: DetectedDie) -> DetectedDie:
        """
        Calculates zone multiplier, score, and out-of-bounds flag for a single die.
        """
        zone = self.find_zone(die.center[0], die.center[1])
        if zone is not None:
            die.zone_name = zone.name
            die.zone_multiplier = zone.multiplier
            die.is_out_of_bounds = (zone.multiplier == 0)
        else:
            die.zone_name = "Out of Bounds"
            die.zone_multiplier = 0
            die.is_out_of_bounds = True

        # Formula: Score = Pip Count * Zone Multiplier (0 if out of bounds)
        die.score = die.pip_count * die.zone_multiplier if not die.is_out_of_bounds else 0
        return die

    def score_dice(
        self,
        dice: Sequence[DetectedDie],
        player_manager: Optional[PlayerProfileManager] = None,
    ) -> DiceScoringResult:
        """
        Evaluates a collection of dice and compiles the full round scoreboard snapshot.
        """
        scored_dice: List[DetectedDie] = []
        out_of_bounds: List[DetectedDie] = []
        unassigned: List[DetectedDie] = []

        # Initialize round score containers per registered player
        player_round_scores: Dict[int, PlayerRoundScore] = {}
        if player_manager is not None:
            for p in player_manager.get_all_profiles():
                player_round_scores[p.player_id] = PlayerRoundScore(
                    player_id=p.player_id,
                    player_name=p.name,
                    color_name=p.color_name,
                    display_bgr=p.display_bgr,
                )

        total_board_score = 0
        active_dice_count = 0

        for d in dice:
            scored = self.score_die(d)
            scored_dice.append(scored)

            if scored.is_out_of_bounds:
                out_of_bounds.append(scored)
            else:
                active_dice_count += 1
                total_board_score += scored.score

            if scored.player_id is None:
                unassigned.append(scored)
            else:
                # Add to player tally
                if scored.player_id not in player_round_scores:
                    player_round_scores[scored.player_id] = PlayerRoundScore(
                        player_id=scored.player_id,
                        player_name=scored.player_name or f"Player {scored.player_id}",
                        color_name=scored.player_color or "Custom",
                        display_bgr=scored.display_bgr,
                    )
                entry = player_round_scores[scored.player_id]
                entry.dice.append(scored)
                entry.total_score += scored.score
                if scored.is_out_of_bounds:
                    entry.out_of_bounds_count += 1
                else:
                    entry.active_dice_count += 1

        # Build sorted leaderboard
        leaderboard_entries: List[Dict[str, Any]] = []
        for p_entry in sorted(player_round_scores.values(), key=lambda e: e.total_score, reverse=True):
            leaderboard_entries.append({
                "rank": len(leaderboard_entries) + 1,
                "player_id": p_entry.player_id,
                "name": p_entry.player_name,
                "color_name": p_entry.color_name,
                "total_score": p_entry.total_score,
                "active_dice": p_entry.active_dice_count,
                "out_of_bounds": p_entry.out_of_bounds_count,
                "display_bgr": list(p_entry.display_bgr),
            })

        player_scores = {p_id: e.total_score for p_id, e in player_round_scores.items()}

        return DiceScoringResult(
            dice=scored_dice,
            player_scores=player_scores,
            player_round_scores=player_round_scores,
            leaderboard=leaderboard_entries,
            out_of_bounds_dice=out_of_bounds,
            unassigned_dice=unassigned,
            total_board_score=total_board_score,
            active_dice_count=active_dice_count,
        )


# =============================================================================
# 7. Dice Detector
# =============================================================================

class DiceDetector:
    """
    Detects physical dice on a calibrated/rectified board view.
    Finds candidates via edge/contrast contours, performs pip counting and color matching.
    """

    def __init__(
        self,
        pip_counter: Optional[D6PipCounter] = None,
        color_sampler: Optional[HSVColorSampler] = None,
        min_die_size: int = 25,
        max_die_size: int = 140,
        min_pip_confidence: float = 0.35,
    ):
        self.pip_counter = pip_counter or D6PipCounter()
        self.color_sampler = color_sampler or HSVColorSampler()
        self.min_die_size = int(min_die_size)
        self.max_die_size = int(max_die_size)
        self.min_pip_confidence = float(min_pip_confidence)

    def extract_candidates(self, board_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Locates candidate die bounding boxes on the board image.
        Uses edge detection, long-line suppression, and color saturation segmentation.
        """
        if board_image is None or board_image.size == 0:
            return []

        gray = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY) if len(board_image.shape) == 3 else board_image

        candidate_boxes: List[Tuple[int, int, int, int]] = []
        scores: List[float] = []

        min_s = self.min_die_size
        max_s = self.max_die_size
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        # Channel 1: Canny edge detection with horizontal & vertical divider line suppression
        edges = cv2.Canny(gray, 40, 140)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Suppress long lines that could connect dice to board tier boundaries
        line_len = int(max_s * 1.5)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_len))
        h_lines = cv2.morphologyEx(closed, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(closed, cv2.MORPH_OPEN, v_kernel)
        cleaned_edges = cv2.subtract(closed, cv2.bitwise_or(h_lines, v_lines))

        # Channel 2: Color Saturation segmentation (dice are highly saturated against board)
        binary_sources = [cleaned_edges]
        if len(board_image.shape) == 3:
            hsv = cv2.cvtColor(board_image, cv2.COLOR_BGR2HSV)
            sat = hsv[:, :, 1]
            _, sat_th = cv2.threshold(sat, 45, 255, cv2.THRESH_BINARY)
            sat_closed = cv2.morphologyEx(sat_th, cv2.MORPH_CLOSE, kernel)
            binary_sources.append(sat_closed)

        for bin_img in binary_sources:
            contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w < min_s or h < min_s or w > max_s or h > max_s:
                    continue

                aspect_ratio = float(w) / float(h)
                if aspect_ratio < 0.65 or aspect_ratio > 1.55:
                    continue

                area = cv2.contourArea(cnt)
                rect_area = float(w * h)
                if rect_area <= 0:
                    continue

                extent = area / rect_area
                # Dice faces fill most of their bounding rectangle
                if extent < 0.40:
                    continue

                candidate_boxes.append((x, y, w, h))
                scores.append(extent)

        # Apply Non-Maximum Suppression to avoid duplicate boxes on same die
        keep_indices = non_maximum_suppression(candidate_boxes, scores, iou_threshold=0.35)
        return [candidate_boxes[i] for i in keep_indices]

    def analyze_candidate(
        self,
        board_image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        die_id: int = 0,
        player_manager: Optional[PlayerProfileManager] = None,
    ) -> Optional[DetectedDie]:
        """
        Analyzes a single bounding box candidate: runs pip counting and color matching.
        Returns DetectedDie if valid pips are found, None otherwise.
        """
        x, y, w, h = bbox
        ih, iw = board_image.shape[:2]

        # Clamp ROI to image boundaries
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(iw, x + w)
        y2 = min(ih, y + h)

        die_roi = board_image[y1:y2, x1:x2]
        if die_roi.size == 0 or die_roi.shape[0] < 10 or die_roi.shape[1] < 10:
            return None

        # 1. Count Pips
        pip_res = self.pip_counter.count_pips(die_roi)
        if pip_res.pip_count < 1 or pip_res.pip_count > 6:
            return None
        if pip_res.confidence < self.min_pip_confidence:
            return None

        # 2. Sample Base Color (with pip exclusion)
        hsv_sampled = self.color_sampler.sample_color_from_roi(die_roi, exclude_pips=True)

        # 3. Match Player Profile
        player_id = None
        player_name = None
        player_color = None
        color_dist = float("inf")
        display_bgr = (255, 255, 255)

        if player_manager is not None:
            matched, dist = player_manager.match_color(hsv_sampled)
            if matched is not None:
                player_id = matched.player_id
                player_name = matched.name
                player_color = matched.color_name
                color_dist = dist
                display_bgr = matched.display_bgr

        # Center point on board plane
        center = (float(x1 + (x2 - x1) / 2.0), float(y1 + (y2 - y1) / 2.0))

        # Absolute pip centers on board
        abs_pips = [(float(x1 + p[0]), float(y1 + p[1])) for p in pip_res.pip_centers]

        return DetectedDie(
            die_id=die_id,
            center=center,
            bbox=(x1, y1, x2 - x1, y2 - y1),
            pip_count=pip_res.pip_count,
            pip_confidence=pip_res.confidence,
            pip_centers=abs_pips,
            player_id=player_id,
            player_name=player_name,
            player_color=player_color,
            color_distance=color_dist,
            hsv_sampled=hsv_sampled,
            display_bgr=display_bgr,
        )

    def detect(
        self,
        board_image: np.ndarray,
        candidate_boxes: Optional[Sequence[Tuple[int, int, int, int]]] = None,
        player_manager: Optional[PlayerProfileManager] = None,
        zone_scorer: Optional[ZoneScorer] = None,
    ) -> List[DetectedDie]:
        """
        Detects all dice on the board image.
        If candidate_boxes is provided, uses them directly; otherwise auto-extracts candidates.
        """
        if board_image is None or board_image.size == 0:
            return []

        boxes = list(candidate_boxes) if candidate_boxes is not None else self.extract_candidates(board_image)

        detected_dice: List[DetectedDie] = []
        die_counter = 1

        for box in boxes:
            die = self.analyze_candidate(board_image, box, die_id=die_counter, player_manager=player_manager)
            if die is not None:
                if zone_scorer is not None:
                    die = zone_scorer.score_die(die)
                detected_dice.append(die)
                die_counter += 1

        return detected_dice


# =============================================================================
# 8. Detection Pipeline & HUD Renderer
# =============================================================================

class DetectionPipeline:
    """
    High-level orchestrator coordinating dice detection, pip counting,
    HSV color sampling, zone scoring, and live HUD overlay rendering.
    """

    def __init__(
        self,
        player_manager: Optional[PlayerProfileManager] = None,
        zone_scorer: Optional[ZoneScorer] = None,
        pip_counter: Optional[D6PipCounter] = None,
        color_sampler: Optional[HSVColorSampler] = None,
        dice_detector: Optional[DiceDetector] = None,
    ):
        self.player_manager = player_manager or PlayerProfileManager.create_default_roster(4)
        self.pip_counter = pip_counter or D6PipCounter()
        self.color_sampler = color_sampler or HSVColorSampler()
        self.zone_scorer = zone_scorer or ZoneScorer()
        self.dice_detector = dice_detector or DiceDetector(
            pip_counter=self.pip_counter, color_sampler=self.color_sampler
        )

        # Storage for active snapshot and manual overrides
        self.current_result: Optional[DiceScoringResult] = None
        self._manual_overrides: Dict[int, Dict[str, Any]] = {}

    def process_frame(
        self,
        frame: np.ndarray,
        candidate_boxes: Optional[Sequence[Tuple[int, int, int, int]]] = None,
    ) -> Tuple[DiceScoringResult, np.ndarray]:
        """
        Runs complete analysis on the board frame.
        Returns: (DiceScoringResult, annotated_hud_frame).
        """
        if frame is None or frame.size == 0:
            raise ValueError("Frame cannot be None or empty.")

        # Detect dice
        dice = self.dice_detector.detect(
            frame,
            candidate_boxes=candidate_boxes,
            player_manager=self.player_manager,
            zone_scorer=self.zone_scorer,
        )

        # Apply manual referee overrides if any exist
        for d in dice:
            if d.die_id in self._manual_overrides:
                override = self._manual_overrides[d.die_id]
                if "pip_count" in override:
                    d.pip_count = int(override["pip_count"])
                if "player_id" in override:
                    pid = int(override["player_id"])
                    d.player_id = pid
                    prof = self.player_manager.get_profile(pid)
                    if prof:
                        d.player_name = prof.name
                        d.player_color = prof.color_name
                        d.display_bgr = prof.display_bgr

        # Score all dice
        result = self.zone_scorer.score_dice(dice, player_manager=self.player_manager)
        self.current_result = result

        # Render HUD overlay
        annotated = self.render_hud(frame, result)
        return result, annotated

    def override_die(
        self, die_id: int, pip_count: Optional[int] = None, player_id: Optional[int] = None
    ) -> None:
        """
        Registers a manual referee override for a die (e.g. if camera misread pips due to glare).
        """
        if die_id not in self._manual_overrides:
            self._manual_overrides[die_id] = {}
        if pip_count is not None:
            if pip_count < 1 or pip_count > 6:
                raise ValueError(f"Pip count must be between 1 and 6, got {pip_count}")
            self._manual_overrides[die_id]["pip_count"] = pip_count
        if player_id is not None:
            self._manual_overrides[die_id]["player_id"] = player_id

    def clear_overrides(self) -> None:
        """Clears all manual overrides."""
        self._manual_overrides.clear()

    def render_hud(
        self,
        frame: np.ndarray,
        result: DiceScoringResult,
        show_zones: bool = True,
        show_leaderboard: bool = True,
    ) -> np.ndarray:
        """
        Renders live broadcast HUD onto the board frame:
        - Zone overlays (if ZoneManager is configured)
        - Highlight rings and pip dots for every die
        - Score pill badges: "[Player Name] | Pips x Mult = Pts"
        - Semi-transparent sidebar leaderboard
        """
        output = frame.copy()

        # 1. Render Zones if ZoneManager is available
        if show_zones and self.zone_scorer.zone_manager is not None:
            output = self.zone_scorer.zone_manager.render_overlay(output, alpha=0.25)

        # 2. Render Dice Highlights & Badges
        for d in result.dice:
            color = d.display_bgr if d.player_id is not None else (200, 200, 200)
            cx, cy = int(d.center[0]), int(d.center[1])
            bx, by, bw, bh = d.bbox
            radius = max(int(max(bw, bh) * 0.55), 18)

            # Circular glow ring around die
            ring_color = (0, 0, 255) if d.is_out_of_bounds else color
            cv2.circle(output, (cx, cy), radius, ring_color, 2, cv2.LINE_AA)

            # Draw center mark
            cv2.circle(output, (cx, cy), 3, (255, 255, 255), -1, cv2.LINE_AA)

            # Construct badge text
            p_name = d.player_name or (f"P{d.player_id}" if d.player_id else "Unassigned")
            if d.is_out_of_bounds:
                badge_text = f"{p_name} | {d.pip_count}p - OUT (0pts)"
            else:
                badge_text = f"{p_name} | {d.pip_count}x{d.zone_multiplier} = {d.score}pts"

            # Badge pill
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.52
            thick = 1
            (tw, th), baseline = cv2.getTextSize(badge_text, font, scale, thick)

            # Position pill above die
            px = int(cx - tw / 2)
            py = max(th + 6, cy - radius - 8)

            # Keep inside frame
            px = max(4, min(output.shape[1] - tw - 8, px))

            pad = 4
            cv2.rectangle(
                output,
                (px - pad, py - th - pad),
                (px + tw + pad, py + baseline + pad),
                (20, 20, 20),
                -1,
            )
            cv2.rectangle(
                output,
                (px - pad, py - th - pad),
                (px + tw + pad, py + baseline + pad),
                ring_color,
                1,
            )
            cv2.putText(output, badge_text, (px, py), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

        # 3. Render Leaderboard Overlay
        if show_leaderboard and result.leaderboard:
            output = self._draw_leaderboard_hud(output, result)

        return output

    def _draw_leaderboard_hud(self, image: np.ndarray, result: DiceScoringResult) -> np.ndarray:
        """Renders leaderboard graphic in top-right corner."""
        ih, iw = image.shape[:2]
        board_w = 260
        board_h = 30 + len(result.leaderboard) * 26 + 10
        margin = 12

        bx = iw - board_w - margin
        by = margin

        overlay = image.copy()
        cv2.rectangle(overlay, (bx, by), (bx + board_w, by + board_h), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.82, image, 0.18, 0, image)

        # Border
        cv2.rectangle(image, (bx, by), (bx + board_w, by + board_h), (80, 80, 80), 1)

        # Title
        cv2.putText(
            image,
            "LIVE LEADERBOARD",
            (bx + 12, by + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 215, 255),
            2,
            cv2.LINE_AA,
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        row_y = by + 46
        for entry in result.leaderboard:
            rank = entry["rank"]
            name = entry["name"]
            score = entry["total_score"]
            dice_cnt = entry["active_dice"]
            color = tuple(entry["display_bgr"])

            # Color dot
            cv2.circle(image, (bx + 18, row_y - 4), 6, color, -1, cv2.LINE_AA)

            text = f"#{rank} {name[:11]}: {score} pts ({dice_cnt}d)"
            cv2.putText(image, text, (bx + 32, row_y), font, 0.46, (240, 240, 240), 1, cv2.LINE_AA)
            row_y += 24

        return image


# =============================================================================
# 9. Synthetic Dice & Board Generators (for Testing & Simulation)
# =============================================================================

def create_synthetic_die(
    size: int = 100,
    pips: int = 1,
    bg_color: Tuple[int, int, int] = (30, 30, 200),
    fg_color: Tuple[int, int, int] = (255, 255, 255),
    angle_deg: float = 0.0,
    noise_std: float = 0.0,
) -> np.ndarray:
    """
    Generates a realistic synthetic D6 die face with specified pip pattern and rotation.
    bg_color and fg_color are BGR tuples.
    """
    if pips < 1 or pips > 6:
        raise ValueError(f"Pips must be between 1 and 6, got {pips}")

    # Render base square die
    die = np.full((size, size, 3), bg_color, dtype=np.uint8)

    # Rounded corners / subtle bevel
    cv2.rectangle(die, (0, 0), (size - 1, size - 1), (max(0, bg_color[0] - 20), max(0, bg_color[1] - 20), max(0, bg_color[2] - 20)), 1)

    r = max(2, int(size * 0.082))
    u = int(size * 0.28)
    c = int(size * 0.50)
    v = int(size * 0.72)

    patterns = {
        1: [(c, c)],
        2: [(u, u), (v, v)],
        3: [(u, u), (c, c), (v, v)],
        4: [(u, u), (v, u), (u, v), (v, v)],
        5: [(u, u), (v, u), (c, c), (u, v), (v, v)],
        6: [(u, u), (u, c), (u, v), (v, u), (v, c), (v, v)],
    }

    for pt in patterns[pips]:
        cv2.circle(die, pt, r, fg_color, -1, cv2.LINE_AA)

    # Rotation if specified
    if abs(angle_deg) > 1e-3:
        m = cv2.getRotationMatrix2D((size / 2.0, size / 2.0), angle_deg, 1.0)
        die = cv2.warpAffine(die, m, (size, size), borderMode=cv2.BORDER_REPLICATE)

    # Optional Gaussian noise
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, die.shape).astype(np.float32)
        die = np.clip(die.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return die


def create_synthetic_board(
    width: int = 800,
    height: int = 1200,
    dice_specs: Optional[Sequence[Dict[str, Any]]] = None,
    zones: Optional[Sequence[ScoringZone]] = None,
    background_color: Tuple[int, int, int] = (190, 210, 220),
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Renders a synthetic board scene with placed dice and optional scoring zones.
    Returns: (board_image, placed_dice_metadata).
    """
    board = np.full((height, width, 3), background_color, dtype=np.uint8)

    # Draw faint board tier lines
    for tier_y in (300, 600, 900):
        cv2.line(board, (0, tier_y), (width, tier_y), (160, 180, 190), 2)

    placed_dice_meta: List[Dict[str, Any]] = []

    if dice_specs:
        for idx, spec in enumerate(dice_specs):
            cx = int(spec.get("cx", 400))
            cy = int(spec.get("cy", 600))
            size = int(spec.get("size", 60))
            pips = int(spec.get("pips", 1))
            bg = tuple(spec.get("bg_color", (30, 30, 200)))
            fg = tuple(spec.get("fg_color", (255, 255, 255)))
            angle = float(spec.get("angle", 0.0))
            player_id = spec.get("player_id")

            die_img = create_synthetic_die(size=size, pips=pips, bg_color=bg, fg_color=fg, angle_deg=angle)

            # Coordinates to paste
            half = size // 2
            x1 = cx - half
            y1 = cy - half
            x2 = x1 + size
            y2 = y1 + size

            if x1 >= 0 and y1 >= 0 and x2 <= width and y2 <= height:
                board[y1:y2, x1:x2] = die_img
                placed_dice_meta.append({
                    "die_id": idx + 1,
                    "center": (float(cx), float(cy)),
                    "bbox": (x1, y1, size, size),
                    "pips": pips,
                    "player_id": player_id,
                    "bg_color": bg,
                })

    return board, placed_dice_meta


# =============================================================================
# 10. Interactive Visual Demo & Live Camera CLI Entrypoint
# =============================================================================

def generate_random_demo_board(
    num_dice: int = 5,
    seed: Optional[int] = None,
    width: int = 800,
    height: int = 1200,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Generates a synthetic board populated with random colored dice."""
    import random

    if seed is not None:
        random.seed(seed)

    player_specs = [
        (1, "Player 1 (Red)", (30, 30, 220)),
        (2, "Player 2 (Blue)", (220, 100, 30)),
        (3, "Player 3 (Green)", (50, 205, 50)),
        (4, "Player 4 (Yellow)", (0, 215, 255)),
    ]

    dice_specs: List[Dict[str, Any]] = []
    for i in range(num_dice):
        pid, name, bg = player_specs[i % len(player_specs)]
        cx = random.randint(140, width - 140)
        cy = random.randint(220, height - 120)
        pips = random.randint(1, 6)
        angle = random.uniform(-40.0, 40.0)
        size = random.randint(52, 62)
        dice_specs.append({
            "cx": cx,
            "cy": cy,
            "size": size,
            "pips": pips,
            "bg_color": bg,
            "fg_color": (255, 255, 255),
            "angle": angle,
            "player_id": pid,
        })

    return create_synthetic_board(width=width, height=height, dice_specs=dice_specs)


def run_demo(
    config_path: str = "zones_config.json",
    headless: bool = False,
    output_path: Optional[str] = None,
) -> None:
    """
    Runs interactive visual demo displaying synthetic dice detection,
    pip counting, player color clustering, zone multiplier scoring, and HUD overlay.
    """
    import random

    print("=" * 68)
    print(" Tumblin' Dice Detection & Pip Scoring — Interactive Demo")
    print("=" * 68)
    print("Demonstrating:")
    print("  1. HSV color clustering & player identification (Red, Blue, Green, Yellow)")
    print("  2. High-accuracy D6 pip pattern recognition (1..6 dots)")
    print("  3. Point-in-polygon zone multiplier scoring (1x..4x, Gutter 0pts)")
    print("  4. Live broadcast HUD overlay with score badges & ranked leaderboard")
    print("-" * 68)
    print("Controls:")
    print("  [Space]     Roll / generate new random dice layout")
    print("  [1..4]      Add a die for Player 1 (Red), 2 (Blue), 3 (Green), or 4 (Yellow)")
    print("  [c]         Clear all dice from the board")
    print("  [s]         Save current annotated HUD frame to screenshot")
    print("  [q / ESC]   Quit demo")
    print("=" * 68)

    # Initialize scoring zones
    zone_scorer = ZoneScorer()
    if os.path.exists(config_path):
        try:
            zone_scorer.set_zones(config_path)
            print(f"[INFO] Loaded {len(zone_scorer.zones)} scoring zones from {config_path}")
        except Exception as e:
            print(f"[WARN] Could not load zones from {config_path}: {e}. Using default.")

    pipeline = DetectionPipeline(zone_scorer=zone_scorer)

    # State: list of active dice specs
    active_specs: List[Dict[str, Any]] = []

    def populate_initial_dice() -> None:
        nonlocal active_specs
        player_specs = [
            (1, (30, 30, 220)),
            (2, (220, 100, 30)),
            (3, (50, 205, 50)),
            (4, (0, 215, 255)),
        ]
        active_specs = [
            # Die in 1x zone
            {"cx": 200, "cy": 500, "size": 56, "pips": 4, "bg_color": player_specs[0][1], "player_id": 1, "angle": 12.0},
            # Die in 2x zone
            {"cx": 480, "cy": 540, "size": 56, "pips": 6, "bg_color": player_specs[1][1], "player_id": 2, "angle": -15.0},
            # Die in 3x zone
            {"cx": 620, "cy": 760, "size": 56, "pips": 5, "bg_color": player_specs[2][1], "player_id": 3, "angle": 5.0},
            # Die in 4x zone
            {"cx": 700, "cy": 1050, "size": 56, "pips": 3, "bg_color": player_specs[3][1], "player_id": 4, "angle": -25.0},
            # Extra die in gutter / out of bounds
            {"cx": 70, "cy": 220, "size": 54, "pips": 2, "bg_color": player_specs[0][1], "player_id": 1, "angle": 30.0},
        ]

    populate_initial_dice()

    def render_current_scene() -> Tuple[DiceScoringResult, np.ndarray]:
        board, _ = create_synthetic_board(width=800, height=1200, dice_specs=active_specs)
        result, hud = pipeline.process_frame(board)

        # Draw interactive banner across the bottom of the HUD
        h, w = hud.shape[:2]
        banner_h = 42
        cv2.rectangle(hud, (0, h - banner_h), (w, h), (20, 20, 20), -1)
        cv2.line(hud, (0, h - banner_h), (w, h - banner_h), (80, 80, 80), 1)
        instructions = "[Space] New Roll | [1..4] Add Die | [c] Clear | [s] Save | [q/ESC] Quit"
        cv2.putText(
            hud,
            instructions,
            (16, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        return result, hud

    # Headless / automated mode
    if headless:
        result, hud = render_current_scene()
        target = output_path or "demo_detection.png"
        cv2.imwrite(target, hud)
        print(f"[INFO] Headless demo complete. Output saved to: {target}")
        print(f"[INFO] Dice detected: {len(result.dice)}, Total score: {result.total_board_score}")
        for entry in result.leaderboard:
            print(f"       Rank {entry.get('rank')}: {entry.get('name')} ({entry.get('color_name')}) -> {entry.get('total_score')} pts")
        return

    window_name = "Tumblin' Dice Detection & Scoring — Interactive Demo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 720, 1000)

    try:
        while True:
            result, hud = render_current_scene()
            cv2.imshow(window_name, hud)
            key = cv2.waitKey(30) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                break
            elif key == 32:  # Space - Randomize board
                num = random.randint(3, 7)
                _, random_meta = generate_random_demo_board(num_dice=num)
                active_specs = []
                player_colors = {
                    1: (30, 30, 220),
                    2: (220, 100, 30),
                    3: (50, 205, 50),
                    4: (0, 215, 255),
                }
                for m in random_meta:
                    pid = m.get("player_id", 1)
                    bx, by, bw, bh = m.get("bbox", (200, 200, 56, 56))
                    active_specs.append({
                        "cx": bx + bw // 2,
                        "cy": by + bh // 2,
                        "size": bw,
                        "pips": m.get("pips", 1),
                        "bg_color": player_colors.get(pid, (30, 30, 220)),
                        "player_id": pid,
                        "angle": random.uniform(-40, 40),
                    })
                print(f"[ACTION] Rolled {len(active_specs)} new random dice onto board.")
            elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
                pid = int(chr(key))
                player_colors = {
                    1: (30, 30, 220),
                    2: (220, 100, 30),
                    3: (50, 205, 50),
                    4: (0, 215, 255),
                }
                cx = random.randint(160, 640)
                cy = random.randint(300, 1100)
                pips = random.randint(1, 6)
                active_specs.append({
                    "cx": cx,
                    "cy": cy,
                    "size": random.randint(52, 60),
                    "pips": pips,
                    "bg_color": player_colors[pid],
                    "player_id": pid,
                    "angle": random.uniform(-35, 35),
                })
                print(f"[ACTION] Added die for Player {pid} (Pips: {pips}) at ({cx}, {cy}).")
            elif key in (ord("c"), ord("C")):
                active_specs.clear()
                print("[ACTION] Cleared all dice from the board.")
            elif key in (ord("s"), ord("S")):
                save_file = "demo_detection_screenshot.png"
                cv2.imwrite(save_file, hud)
                print(f"[ACTION] Saved screenshot to {os.path.abspath(save_file)}")
    finally:
        cv2.destroyAllWindows()


def run_camera(
    camera_idx: int = 0,
    config_path: str = "zones_config.json",
    calibrate: bool = False,
) -> None:
    """
    Runs live dice detection and scoring from an attached camera device.
    Supports Windows DirectShow backend to prevent MSMF video frame errors.
    """
    import sys

    print("=" * 68)
    print(f" Tumblin' Dice Live Camera Mode (Device Index: {camera_idx})")
    print("=" * 68)
    if calibrate:
        print(" [CALIBRATION MODE]")
        print(" Click on any die in the video feed to sample its HSV color profile.")
    print(" Press [q / ESC] to exit camera feed.")
    print("=" * 68)

    # Initialize camera with DirectShow on Windows for reliability
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
        print("Please check your camera connection or try another device index (e.g. --camera 1).")
        return

    # Initialize zones and pipeline
    zone_scorer = ZoneScorer()
    if os.path.exists(config_path):
        try:
            zone_scorer.set_zones(config_path)
            print(f"[INFO] Loaded zones from {config_path}")
        except Exception as e:
            print(f"[WARN] Could not load zones from {config_path}: {e}")

    pipeline = DetectionPipeline(zone_scorer=zone_scorer)
    window_name = f"Tumblin' Dice Live Camera — Device {camera_idx}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    last_click: Optional[Tuple[int, int]] = None

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        nonlocal last_click
        if event == cv2.EVENT_LBUTTONDOWN and calibrate:
            last_click = (x, y)

    cv2.setMouseCallback(window_name, on_mouse)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] Failed to grab frame from camera.")
                break

            # Handle calibration click
            if last_click is not None and calibrate:
                cx, cy = last_click
                last_click = None
                # Sample 20x20 patch around click
                x1 = max(0, cx - 10)
                y1 = max(0, cy - 10)
                x2 = min(frame.shape[1], cx + 10)
                y2 = min(frame.shape[0], cy + 10)
                patch = frame[y1:y2, x1:x2]
                if patch.size > 0:
                    sampler = HSVColorSampler()
                    hsv_stats = sampler.sample_color_from_roi(patch, exclude_pips=False)
                    print(f"[CALIBRATE] Sampled at ({cx}, {cy}): Hue={hsv_stats.mean_hue:.1f}, Sat={hsv_stats.mean_sat:.1f}, Val={hsv_stats.mean_val:.1f}")

            # Process frame through pipeline
            try:
                result, hud = pipeline.process_frame(frame)
            except Exception as e:
                hud = frame.copy()
                cv2.putText(hud, f"Processing Error: {e}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow(window_name, hud)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    """CLI entry point for Tumblin' Dice detection module."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Tumblin' Dice Computer Vision Detection & Scoring Module"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run interactive visual demo displaying synthetic dice detection, pip counting, and scoring HUD",
    )
    parser.add_argument(
        "--camera",
        type=int,
        nargs="?",
        const=0,
        default=None,
        help="Run live dice detection using webcam device index (default: 0)",
    )
    parser.add_argument(
        "--calibrate-colors",
        action="store_true",
        help="Enable interactive color calibration mode when running with --camera",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="zones_config.json",
        help="Path to scoring zones JSON configuration file (default: zones_config.json)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run demo in headless mode (no GUI window, writes output image and logs to console)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Destination path for output image when running with --headless",
    )

    args = parser.parse_args()

    if args.camera is not None:
        run_camera(
            camera_idx=args.camera,
            config_path=args.config,
            calibrate=args.calibrate_colors,
        )
    elif args.demo:
        run_demo(
            config_path=args.config,
            headless=args.headless,
            output_path=args.output,
        )
    else:
        # Default to interactive demo mode
        print("[INFO] No mode specified. Defaulting to interactive simulation demo (--demo).")
        run_demo(
            config_path=args.config,
            headless=args.headless,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()

