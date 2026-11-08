"""
test_motion.py - Comprehensive Unit Tests for Tumblin' Dice Motion Detector

Verification test command:
python -m unittest test_motion.py
"""

from __future__ import annotations

import unittest
from typing import List, Tuple

import cv2
import numpy as np

from motion import (
    DEFAULT_ACCUMULATION_FRAMES,
    DEFAULT_BLUR_KSIZE,
    DEFAULT_DIFF_THRESHOLD,
    DEFAULT_DILATION_ITERATIONS,
    DEFAULT_MIN_CONTOUR_AREA,
    DEFAULT_MOTION_THRESHOLD,
    DEFAULT_QUIESCENCE_DURATION,
    MotionConfig,
    MotionDetector,
    MotionResult,
    MotionState,
)


def create_blank_frame(
    width: int = 640,
    height: int = 480,
    color: Tuple[int, int, int] = (40, 40, 40),
) -> np.ndarray:
    """Helper to create a solid BGR test frame."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = color
    return frame


def draw_test_die(
    frame: np.ndarray,
    center: Tuple[int, int],
    size: int = 30,
    color: Tuple[int, int, int] = (220, 220, 220),
) -> np.ndarray:
    """Helper to draw a simulated square die with contrasting pip."""
    img = frame.copy()
    cx, cy = center
    half = size // 2
    cv2.rectangle(img, (cx - half, cy - half), (cx + half, cy + half), color, -1)
    # Contrasting pip
    cv2.circle(img, (cx, cy), max(2, size // 8), (20, 20, 20), -1)
    return img


class TestMotionConfig(unittest.TestCase):
    """Tests for MotionConfig parameter validation, defaults, and serialization."""

    def test_default_config(self):
        cfg = MotionConfig()
        self.assertEqual(cfg.quiescence_duration, DEFAULT_QUIESCENCE_DURATION)
        self.assertEqual(cfg.quiescence_duration, 1.2)
        self.assertEqual(cfg.motion_threshold, DEFAULT_MOTION_THRESHOLD)
        self.assertEqual(cfg.diff_threshold, DEFAULT_DIFF_THRESHOLD)
        self.assertEqual(cfg.blur_ksize, DEFAULT_BLUR_KSIZE)
        self.assertEqual(cfg.min_contour_area, DEFAULT_MIN_CONTOUR_AREA)
        self.assertEqual(cfg.accumulation_frames, DEFAULT_ACCUMULATION_FRAMES)
        self.assertEqual(cfg.dilation_iterations, DEFAULT_DILATION_ITERATIONS)
        self.assertIsNone(cfg.roi)
        self.assertIsNone(cfg.roi_polygon)

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            MotionConfig(quiescence_duration=0.0)
        with self.assertRaises(ValueError):
            MotionConfig(quiescence_duration=-1.0)
        with self.assertRaises(ValueError):
            MotionConfig(motion_threshold=-10.0)
        with self.assertRaises(ValueError):
            MotionConfig(diff_threshold=256)
        with self.assertRaises(ValueError):
            MotionConfig(diff_threshold=-1)
        with self.assertRaises(ValueError):
            MotionConfig(blur_ksize=(20, 20))  # Even kernel size
        with self.assertRaises(ValueError):
            MotionConfig(accumulation_frames=0)
        with self.assertRaises(ValueError):
            MotionConfig(dilation_iterations=-1)
        with self.assertRaises(ValueError):
            MotionConfig(roi=(10, 10, -5, 20))
        with self.assertRaises(ValueError):
            MotionConfig(roi_polygon=[(0.0, 0.0), (10.0, 10.0)])  # Only 2 points

    def test_serialization_round_trip(self):
        cfg = MotionConfig(
            quiescence_duration=1.5,
            motion_threshold=350.0,
            diff_threshold=30,
            blur_ksize=(15, 15),
            min_contour_area=40.0,
            accumulation_frames=5,
            dilation_iterations=3,
            gaussian_sigma=1.2,
            roi=(50, 50, 200, 300),
            roi_polygon=[(10.0, 10.0), (100.0, 10.0), (100.0, 100.0), (10.0, 100.0)],
        )
        d = cfg.to_dict()
        cfg_loaded = MotionConfig.from_dict(d)
        self.assertEqual(cfg.quiescence_duration, cfg_loaded.quiescence_duration)
        self.assertEqual(cfg.motion_threshold, cfg_loaded.motion_threshold)
        self.assertEqual(cfg.diff_threshold, cfg_loaded.diff_threshold)
        self.assertEqual(cfg.blur_ksize, cfg_loaded.blur_ksize)
        self.assertEqual(cfg.min_contour_area, cfg_loaded.min_contour_area)
        self.assertEqual(cfg.accumulation_frames, cfg_loaded.accumulation_frames)
        self.assertEqual(cfg.dilation_iterations, cfg_loaded.dilation_iterations)
        self.assertEqual(cfg.roi, cfg_loaded.roi)
        self.assertEqual(cfg.roi_polygon, cfg_loaded.roi_polygon)


class TestMotionState(unittest.TestCase):
    """Tests for MotionState enum."""

    def test_enum_values(self):
        self.assertEqual(MotionState.IDLE.value, "idle")
        self.assertEqual(MotionState.MOVING.value, "moving")
        self.assertEqual(MotionState.SETTLING.value, "settling")
        self.assertEqual(MotionState.SETTLED.value, "settled")
        # Direct string equality thanks to (str, Enum)
        self.assertEqual(MotionState.IDLE, "idle")
        self.assertEqual(MotionState.MOVING, "moving")
        self.assertEqual(MotionState.SETTLING, "settling")
        self.assertEqual(MotionState.SETTLED, "settled")


class TestMotionResult(unittest.TestCase):
    """Tests for MotionResult container attributes and properties."""

    def test_result_properties(self):
        res_moving = MotionResult(
            state=MotionState.MOVING,
            movement_score=850.0,
            motion_detected=True,
            is_settled=False,
            is_scoreboard_frozen=True,
            just_settled=False,
            stillness_duration=0.0,
            quiescence_progress=0.0,
            timestamp=1.0,
        )
        self.assertTrue(res_moving.is_moving)
        self.assertFalse(res_moving.is_settling)
        self.assertTrue(res_moving.is_scoreboard_frozen)
        self.assertFalse(res_moving.just_settled)

        res_settling = MotionResult(
            state=MotionState.SETTLING,
            movement_score=10.0,
            motion_detected=False,
            is_settled=False,
            is_scoreboard_frozen=True,
            just_settled=False,
            stillness_duration=0.6,
            quiescence_progress=0.5,
            timestamp=1.6,
        )
        self.assertFalse(res_settling.is_moving)
        self.assertTrue(res_settling.is_settling)
        self.assertTrue(res_settling.is_scoreboard_frozen)

        res_settled = MotionResult(
            state=MotionState.SETTLED,
            movement_score=0.0,
            motion_detected=False,
            is_settled=True,
            is_scoreboard_frozen=False,
            just_settled=True,
            stillness_duration=1.2,
            quiescence_progress=1.0,
            timestamp=2.2,
        )
        self.assertFalse(res_settled.is_moving)
        self.assertFalse(res_settled.is_settling)
        self.assertTrue(res_settled.is_settled)
        self.assertFalse(res_settled.is_scoreboard_frozen)
        self.assertTrue(res_settled.just_settled)


class TestFrameDifferencing(unittest.TestCase):
    """Tests for OpenCV frame differencing, noise filtering, and contour extraction."""

    def setUp(self):
        self.detector = MotionDetector(
            MotionConfig(
                quiescence_duration=1.2,
                motion_threshold=200.0,
                diff_threshold=25,
                blur_ksize=(15, 15),
                min_contour_area=40.0,
                dilation_iterations=1,
            )
        )

    def test_initial_frame_produces_zero_movement(self):
        f1 = create_blank_frame()
        res = self.detector.update(f1, timestamp=0.0)
        self.assertEqual(res.movement_score, 0.0)
        self.assertFalse(res.motion_detected)
        self.assertEqual(res.state, MotionState.IDLE)
        self.assertFalse(res.is_scoreboard_frozen)
        self.assertFalse(res.just_settled)

    def test_identical_consecutive_frames_remain_still(self):
        f = create_blank_frame()
        self.detector.update(f, timestamp=0.0)
        res = self.detector.update(f, timestamp=0.1)
        self.assertEqual(res.movement_score, 0.0)
        self.assertFalse(res.motion_detected)
        self.assertEqual(res.state, MotionState.IDLE)
        self.assertFalse(res.is_scoreboard_frozen)

    def test_moving_object_detected(self):
        f1 = create_blank_frame()
        # Draw die at (100, 100)
        f1 = draw_test_die(f1, (100, 100), size=40)
        self.detector.update(f1, timestamp=0.0)

        # Move die to (180, 180)
        f2 = create_blank_frame()
        f2 = draw_test_die(f2, (180, 180), size=40)
        res = self.detector.update(f2, timestamp=0.1)

        self.assertGreater(res.movement_score, 200.0)
        self.assertTrue(res.motion_detected)
        self.assertEqual(res.state, MotionState.MOVING)
        self.assertTrue(res.is_scoreboard_frozen)
        self.assertGreater(len(res.bounding_boxes), 0)
        self.assertGreater(len(res.motion_contours), 0)

    def test_gaussian_blur_suppresses_pixel_noise(self):
        f1 = create_blank_frame()
        self.detector.update(f1, timestamp=0.0)

        # Inject single-pixel isolated noise spikes into f2
        f2 = f1.copy()
        np.random.seed(42)
        noise_coords_y = np.random.randint(0, 480, size=15)
        noise_coords_x = np.random.randint(0, 640, size=15)
        for y, x in zip(noise_coords_y, noise_coords_x):
            f2[y, x] = (255, 255, 255)

        res = self.detector.update(f2, timestamp=0.1)
        # Because of Gaussian blur (15, 15) and diff threshold 25, single isolated noise is smoothed out
        self.assertFalse(res.motion_detected)
        self.assertEqual(len(res.bounding_boxes), 0)

    def test_diff_threshold_sensitivity(self):
        # Sensitive detector
        det_sensitive = MotionDetector(
            MotionConfig(diff_threshold=10, motion_threshold=50.0, min_contour_area=10.0)
        )
        # Insensitive detector
        det_insensitive = MotionDetector(
            MotionConfig(diff_threshold=80, motion_threshold=50.0, min_contour_area=10.0)
        )

        base = create_blank_frame(color=(50, 50, 50))
        det_sensitive.update(base, timestamp=0.0)
        det_insensitive.update(base, timestamp=0.0)

        # Subtle brightness change of +30 (between 10 and 80)
        subtle = base.copy()
        cv2.rectangle(subtle, (100, 100), (160, 160), (80, 80, 80), -1)

        res_sens = det_sensitive.update(subtle, timestamp=0.1)
        res_insens = det_insensitive.update(subtle, timestamp=0.1)

        self.assertTrue(res_sens.motion_detected)
        self.assertFalse(res_insens.motion_detected)

    def test_different_frame_formats(self):
        detector = MotionDetector()
        # BGR (3 channels)
        bgr = np.zeros((200, 200, 3), dtype=np.uint8)
        res1 = detector.update(bgr, timestamp=0.0)
        self.assertIsNotNone(res1)

        # Grayscale (2D array)
        gray = np.zeros((200, 200), dtype=np.uint8)
        res2 = detector.update(gray, timestamp=0.1)
        self.assertIsNotNone(res2)

        # BGRA (4 channels)
        bgra = np.zeros((200, 200, 4), dtype=np.uint8)
        res3 = detector.update(bgra, timestamp=0.2)
        self.assertIsNotNone(res3)

        # Single channel 3D array (H, W, 1)
        gray3d = np.zeros((200, 200, 1), dtype=np.uint8)
        res4 = detector.update(gray3d, timestamp=0.3)
        self.assertIsNotNone(res4)

    def test_invalid_frame_inputs(self):
        detector = MotionDetector()
        with self.assertRaises(ValueError):
            detector.update(None)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            detector.update(np.array([]))  # Empty array
        with self.assertRaises(ValueError):
            detector.update(np.zeros((10, 10, 5), dtype=np.uint8))  # 5 channels unsupported

    def test_resolution_change_handled_cleanly(self):
        detector = MotionDetector()
        f_small = np.zeros((240, 320, 3), dtype=np.uint8)
        f_large = np.zeros((480, 640, 3), dtype=np.uint8)

        detector.update(f_small, timestamp=0.0)
        # Switching resolution resets previous buffer without crashing
        res = detector.update(f_large, timestamp=0.1)
        self.assertEqual(res.movement_score, 0.0)
        self.assertEqual(res.state, MotionState.IDLE)


class TestQuiescenceStateTransitions(unittest.TestCase):
    """
    Tests for complete lifecycle of motion and quiescence gating:
    IDLE -> MOVING (freeze scoreboard) -> SETTLING (hold freeze) -> SETTLED (>1.2s stillness, unfreeze scoreboard & trigger scoring).
    """

    def setUp(self):
        self.quiescence = 1.2
        self.detector = MotionDetector(
            MotionConfig(
                quiescence_duration=self.quiescence,
                motion_threshold=200.0,
                diff_threshold=25,
                blur_ksize=(15, 15),
            )
        )

    def test_full_roll_and_settle_lifecycle(self):
        blank = create_blank_frame()

        # Step 1: Initial idle frame
        res = self.detector.update(blank, timestamp=0.0)
        self.assertEqual(res.state, MotionState.IDLE)
        self.assertFalse(res.is_scoreboard_frozen)
        self.assertFalse(res.just_settled)

        # Step 2: Die flicked and rolls across board (Motion active)
        # Frame with moving die at position 1
        f1 = draw_test_die(blank, (100, 200), size=50)
        res = self.detector.update(f1, timestamp=0.1)
        self.assertEqual(res.state, MotionState.MOVING)
        self.assertTrue(res.is_scoreboard_frozen)
        self.assertFalse(res.just_settled)
        self.assertEqual(res.stillness_duration, 0.0)

        # Frame with moving die at position 2 (still moving)
        f2 = draw_test_die(blank, (220, 200), size=50)
        res = self.detector.update(f2, timestamp=0.4)
        self.assertEqual(res.state, MotionState.MOVING)
        self.assertTrue(res.is_scoreboard_frozen)
        self.assertFalse(res.just_settled)

        # Step 3: Die comes to rest at (220, 200) -> Motion stops!
        # Consecutive identical frame at t=0.5 -> triggers transition to SETTLING
        f_rest = f2.copy()
        res = self.detector.update(f_rest, timestamp=0.5)
        self.assertEqual(res.state, MotionState.SETTLING)
        self.assertTrue(res.is_scoreboard_frozen, "Scoreboard MUST remain frozen during settling")
        self.assertFalse(res.just_settled)
        self.assertEqual(res.stillness_duration, 0.0)
        self.assertEqual(res.quiescence_progress, 0.0)

        # Step 4: 0.5s of stillness elapsed (t=1.0) -> Still settling (< 1.2s)
        res = self.detector.update(f_rest, timestamp=1.0)
        self.assertEqual(res.state, MotionState.SETTLING)
        self.assertTrue(res.is_scoreboard_frozen, "Scoreboard MUST remain frozen while stillness < 1.2s")
        self.assertAlmostEqual(res.stillness_duration, 0.5, places=3)
        self.assertAlmostEqual(res.quiescence_progress, 0.5 / 1.2, places=3)
        self.assertFalse(res.just_settled)

        # Step 5: 1.15s of stillness elapsed (t=1.65) -> Still settling (< 1.2s)
        res = self.detector.update(f_rest, timestamp=1.65)
        self.assertEqual(res.state, MotionState.SETTLING)
        self.assertTrue(res.is_scoreboard_frozen)
        self.assertAlmostEqual(res.stillness_duration, 1.15, places=3)
        self.assertFalse(res.just_settled)

        # Step 6: 1.25s of continuous stillness elapsed (t=1.75) -> Quiescence threshold >1.2s crossed!
        res = self.detector.update(f_rest, timestamp=1.75)
        self.assertEqual(res.state, MotionState.SETTLED)
        self.assertFalse(res.is_scoreboard_frozen, "Scoreboard unfreezes upon settle")
        self.assertTrue(res.just_settled, "just_settled MUST be True on settle frame")
        self.assertGreaterEqual(res.stillness_duration, 1.2)
        self.assertEqual(res.quiescence_progress, 1.0)

        # Step 7: Subsequent still frames at t=2.0 -> Remains SETTLED, but just_settled is False (edge-triggered)
        res = self.detector.update(f_rest, timestamp=2.0)
        self.assertEqual(res.state, MotionState.SETTLED)
        self.assertFalse(res.is_scoreboard_frozen)
        self.assertFalse(res.just_settled, "just_settled MUST be False after the initial transition")

    def test_quiescence_timer_interrupted_by_new_motion(self):
        """If dice wobble or secondary roll occurs before 1.2s, stillness timer resets."""
        blank = create_blank_frame()
        f_die1 = draw_test_die(blank, (100, 100), size=40)
        f_die2 = draw_test_die(blank, (150, 100), size=40)

        self.detector.update(blank, timestamp=0.0)
        self.detector.update(f_die1, timestamp=0.1)  # MOVING
        self.detector.update(f_die2, timestamp=0.2)  # MOVING

        # Die stops at t=0.3 -> starts settling
        self.detector.update(f_die2, timestamp=0.3)
        self.assertEqual(self.detector.state, MotionState.SETTLING)

        # Still until t=1.2 (stillness = 0.9s, which is < 1.2s)
        res = self.detector.update(f_die2, timestamp=1.2)
        self.assertEqual(res.state, MotionState.SETTLING)
        self.assertAlmostEqual(res.stillness_duration, 0.9, places=3)
        self.assertTrue(res.is_scoreboard_frozen)

        # At t=1.3, another die moves (wobble or rebound)!
        f_die3 = draw_test_die(blank, (250, 100), size=40)
        res = self.detector.update(f_die3, timestamp=1.3)

        # State MUST revert to MOVING, timer resets to 0.0, scoreboard remains frozen
        self.assertEqual(res.state, MotionState.MOVING)
        self.assertTrue(res.is_scoreboard_frozen)
        self.assertEqual(res.stillness_duration, 0.0)
        self.assertFalse(res.just_settled)

        # Now dice stop moving at t=1.5
        self.detector.update(f_die3, timestamp=1.5)
        self.assertEqual(self.detector.state, MotionState.SETTLING)

        # At t=2.6 (stillness = 1.1s < 1.2s from 1.5), still not settled
        res = self.detector.update(f_die3, timestamp=2.6)
        self.assertEqual(res.state, MotionState.SETTLING)
        self.assertFalse(res.just_settled)

        # At t=2.75 (stillness = 1.25s >= 1.2s), NOW settles!
        res = self.detector.update(f_die3, timestamp=2.75)
        self.assertEqual(res.state, MotionState.SETTLED)
        self.assertTrue(res.just_settled)
        self.assertFalse(res.is_scoreboard_frozen)

    def test_custom_quiescence_duration(self):
        # Configure custom 0.5s quiescence
        detector = MotionDetector(MotionConfig(quiescence_duration=0.5, motion_threshold=100.0))
        blank = create_blank_frame()
        f_moving = draw_test_die(blank, (150, 150), size=30)

        detector.update(blank, timestamp=0.0)
        detector.update(f_moving, timestamp=0.1)  # MOVING
        detector.update(f_moving, timestamp=0.2)  # SETTLING begins at 0.2

        # At t=0.6, stillness is 0.4s (< 0.5s)
        res = detector.update(f_moving, timestamp=0.6)
        self.assertEqual(res.state, MotionState.SETTLING)
        self.assertFalse(res.just_settled)

        # At t=0.72, stillness is 0.52s (> 0.5s)
        res = detector.update(f_moving, timestamp=0.72)
        self.assertEqual(res.state, MotionState.SETTLED)
        self.assertTrue(res.just_settled)


class TestAccumulationBuffer(unittest.TestCase):
    """Tests for multi-frame accumulation buffer preventing false settle triggers."""

    def test_accumulation_retains_motion_signal(self):
        # 3-frame accumulation buffer
        detector = MotionDetector(
            MotionConfig(
                accumulation_frames=3,
                motion_threshold=100.0,
                diff_threshold=20,
            )
        )
        blank = create_blank_frame()
        f1 = draw_test_die(blank, (50, 50), size=30)
        f2 = draw_test_die(blank, (100, 50), size=30)

        detector.update(blank, timestamp=0.0)
        res1 = detector.update(f1, timestamp=0.033)
        res2 = detector.update(f2, timestamp=0.066)

        self.assertTrue(res1.motion_detected)
        self.assertTrue(res2.motion_detected)
        # History deque holds frames
        self.assertEqual(len(detector._history_masks), 2)

    def test_single_frame_mode_disables_accumulation(self):
        detector = MotionDetector(
            MotionConfig(accumulation_frames=1, motion_threshold=100.0)
        )
        blank = create_blank_frame()
        detector.update(blank, timestamp=0.0)
        self.assertEqual(len(detector._history_masks), 0)


class TestROIAndMasking(unittest.TestCase):
    """Tests for restricting motion detection to a defined board ROI or irregular polygon."""

    def test_rectangular_roi_ignores_outside_motion(self):
        detector = MotionDetector(
            MotionConfig(
                motion_threshold=150.0,
                roi=(100, 100, 200, 200),  # Box from x=100..300, y=100..300
            )
        )
        blank = create_blank_frame(width=640, height=480)
        detector.update(blank, timestamp=0.0)

        # Motion happens at (500, 400) -> OUTSIDE the ROI box
        f_outside = draw_test_die(blank, (500, 400), size=40)
        res_out = detector.update(f_outside, timestamp=0.1)
        self.assertEqual(res_out.movement_score, 0.0)
        self.assertFalse(res_out.motion_detected)

        # Motion happens at (150, 150) -> INSIDE the ROI box
        f_inside = draw_test_die(f_outside, (150, 150), size=40)
        res_in = detector.update(f_inside, timestamp=0.2)
        self.assertGreater(res_in.movement_score, 150.0)
        self.assertTrue(res_in.motion_detected)

    def test_polygon_roi_mask(self):
        # Triangular ROI
        triangle = [(100.0, 100.0), (300.0, 100.0), (200.0, 300.0)]
        detector = MotionDetector(MotionConfig(motion_threshold=150.0, roi_polygon=triangle))

        blank = create_blank_frame(width=640, height=480)
        detector.update(blank, timestamp=0.0)

        # Motion outside triangle (e.g. at 50, 50)
        f_outside = draw_test_die(blank, (50, 50), size=40)
        res_out = detector.update(f_outside, timestamp=0.1)
        self.assertEqual(res_out.movement_score, 0.0)

        # Motion inside triangle (e.g. at 200, 150)
        f_inside = draw_test_die(f_outside, (200, 150), size=40)
        res_in = detector.update(f_inside, timestamp=0.2)
        self.assertGreater(res_in.movement_score, 150.0)
        self.assertTrue(res_in.motion_detected)

    def test_dynamic_roi_helpers(self):
        detector = MotionDetector()
        detector.set_roi(10, 10, 100, 100)
        self.assertEqual(detector.config.roi, (10, 10, 100, 100))
        self.assertIsNone(detector.config.roi_polygon)

        detector.set_roi_polygon([(0, 0), (50, 0), (50, 50)])
        self.assertEqual(len(detector.config.roi_polygon), 3)
        self.assertIsNone(detector.config.roi)

        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:50, 20:50] = 255
        detector.set_mask(mask)
        self.assertIsNotNone(detector._roi_mask)

        detector.clear_roi()
        self.assertIsNone(detector.config.roi)
        self.assertIsNone(detector.config.roi_polygon)
        self.assertIsNone(detector._roi_mask)

    def test_invalid_roi_helpers_raise(self):
        detector = MotionDetector()
        with self.assertRaises(ValueError):
            detector.set_roi(0, 0, -10, 20)
        with self.assertRaises(ValueError):
            detector.set_roi_polygon([(0, 0), (1, 1)])  # < 3 points
        with self.assertRaises(ValueError):
            detector.set_mask(np.zeros((10, 10, 3), dtype=np.uint8))  # 3D mask invalid


class TestFailsafesAndManualOverrides(unittest.TestCase):
    """Tests for force_settle, force_motion, and reset methods."""

    def test_force_settle_unfreezes_and_triggers_settle(self):
        detector = MotionDetector()
        settle_events: List[MotionResult] = []
        detector.register_on_settle(lambda r: settle_events.append(r))

        # Put detector into moving state
        blank = create_blank_frame()
        detector.update(blank, timestamp=0.0)
        f_mov = draw_test_die(blank, (100, 100), size=40)
        detector.update(f_mov, timestamp=0.1)
        self.assertEqual(detector.state, MotionState.MOVING)
        self.assertTrue(detector.is_scoreboard_frozen)

        # Trigger manual force settle ("Next Turn / Lock Frame" button)
        result = detector.force_settle(timestamp=0.5)
        self.assertEqual(detector.state, MotionState.SETTLED)
        self.assertFalse(detector.is_scoreboard_frozen)
        self.assertTrue(detector.just_settled)
        self.assertEqual(len(settle_events), 1)
        self.assertEqual(settle_events[0], result)

    def test_force_motion_freezes_scoreboard(self):
        detector = MotionDetector()
        motion_events: List[MotionResult] = []
        detector.register_on_motion_start(lambda r: motion_events.append(r))

        res = detector.force_motion(timestamp=1.0)
        self.assertEqual(detector.state, MotionState.MOVING)
        self.assertTrue(detector.is_scoreboard_frozen)
        self.assertFalse(detector.just_settled)
        self.assertEqual(len(motion_events), 1)

    def test_reset_restores_clean_idle_state(self):
        detector = MotionDetector()
        detector.force_motion(timestamp=1.0)
        self.assertEqual(detector.state, MotionState.MOVING)

        detector.reset()
        self.assertEqual(detector.state, MotionState.IDLE)
        self.assertFalse(detector.is_scoreboard_frozen)
        self.assertFalse(detector.just_settled)
        self.assertIsNone(detector._prev_blurred)
        self.assertEqual(len(detector._history_masks), 0)


class TestCallbacks(unittest.TestCase):
    """Tests for on_settle and on_motion_start event dispatch."""

    def test_on_motion_start_and_on_settle_callbacks(self):
        detector = MotionDetector(MotionConfig(quiescence_duration=1.2, motion_threshold=150.0))

        start_calls: List[float] = []
        settle_calls: List[float] = []

        detector.register_on_motion_start(lambda r: start_calls.append(r.timestamp))
        detector.register_on_settle(lambda r: settle_calls.append(r.timestamp))

        blank = create_blank_frame()
        detector.update(blank, timestamp=0.0)
        self.assertEqual(len(start_calls), 0)
        self.assertEqual(len(settle_calls), 0)

        # Motion starts at t=0.1
        f1 = draw_test_die(blank, (100, 100), size=40)
        detector.update(f1, timestamp=0.1)
        self.assertEqual(len(start_calls), 1)
        self.assertEqual(start_calls[0], 0.1)
        self.assertEqual(len(settle_calls), 0)

        # Motion continues at t=0.2 (should not re-trigger start callback)
        f2 = draw_test_die(blank, (150, 100), size=40)
        detector.update(f2, timestamp=0.2)
        self.assertEqual(len(start_calls), 1)

        # Motion stops at t=0.3
        detector.update(f2, timestamp=0.3)
        self.assertEqual(len(settle_calls), 0)

        # Still at t=1.0 (0.7s stillness)
        detector.update(f2, timestamp=1.0)
        self.assertEqual(len(settle_calls), 0)

        # Settle at t=1.55 (1.25s stillness >= 1.2s)
        detector.update(f2, timestamp=1.55)
        self.assertEqual(len(settle_calls), 1)
        self.assertEqual(settle_calls[0], 1.55)

        # Continues still at t=2.0 (should not fire again)
        detector.update(f2, timestamp=2.0)
        self.assertEqual(len(settle_calls), 1)

    def test_callback_type_error(self):
        detector = MotionDetector()
        with self.assertRaises(TypeError):
            detector.register_on_settle("not_callable")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            detector.register_on_motion_start(None)  # type: ignore[arg-type]

    def test_faulty_callback_does_not_crash_detector(self):
        detector = MotionDetector()

        def bad_cb(result):
            raise RuntimeError("Failure in callback")

        detector.register_on_settle(bad_cb)
        with self.assertLogs("motion", level="ERROR") as cm:
            detector.force_settle()
        self.assertEqual(detector.state, MotionState.SETTLED)
        self.assertTrue(any("Failure in callback" in msg for msg in cm.output))


class TestHUDOverlay(unittest.TestCase):
    """Tests for drawing HUD overlay in all motion states."""

    def setUp(self):
        self.detector = MotionDetector(
            MotionConfig(roi=(50, 50, 200, 200), motion_threshold=200.0)
        )
        self.frame = create_blank_frame(640, 480)

    def test_draw_overlay_idle(self):
        res = self.detector.update(self.frame, timestamp=0.0)
        annotated = self.detector.draw_overlay(self.frame, res)
        self.assertEqual(annotated.shape, self.frame.shape)
        self.assertEqual(annotated.dtype, np.uint8)

    def test_draw_overlay_moving(self):
        self.detector.update(self.frame, timestamp=0.0)
        f_mov = draw_test_die(self.frame, (100, 100), size=40)
        res = self.detector.update(f_mov, timestamp=0.1)
        annotated = self.detector.draw_overlay(f_mov, res)
        self.assertEqual(annotated.shape, f_mov.shape)

    def test_draw_overlay_settling(self):
        self.detector.update(self.frame, timestamp=0.0)
        f_mov = draw_test_die(self.frame, (100, 100), size=40)
        self.detector.update(f_mov, timestamp=0.1)
        # Settle frame
        res = self.detector.update(f_mov, timestamp=0.5)
        annotated = self.detector.draw_overlay(f_mov, res)
        self.assertEqual(annotated.shape, f_mov.shape)

    def test_draw_overlay_settled(self):
        res = self.detector.force_settle(timestamp=1.0)
        annotated = self.detector.draw_overlay(self.frame, res)
        self.assertEqual(annotated.shape, self.frame.shape)

    def test_draw_overlay_options(self):
        res = self.detector.update(self.frame, timestamp=0.0)
        # Turn off all elements
        annotated = self.detector.draw_overlay(
            self.frame,
            res,
            show_badge=False,
            show_contours=False,
            show_meter=False,
            show_roi=False,
        )
        self.assertEqual(annotated.shape, self.frame.shape)

    def test_draw_overlay_polygon_roi(self):
        self.detector.set_roi_polygon([(50, 50), (150, 50), (100, 150)])
        annotated = self.detector.draw_overlay(self.frame)
        self.assertEqual(annotated.shape, self.frame.shape)

    def test_draw_overlay_empty_frame(self):
        empty = np.array([])
        out = self.detector.draw_overlay(empty)
        self.assertEqual(out.size, 0)


class TestInteroperabilityWithCalibration(unittest.TestCase):
    """
    Tests interoperability with calibration module (e.g. ScoringZone polygon integration).
    """

    def test_polygon_from_calibration_scoring_zone(self):
        try:
            from calibration import ScoringZone
        except ImportError:
            self.skipTest("calibration.py not found in workspace")

        zone = ScoringZone(
            name="Zone 4 (4x)",
            multiplier=4,
            points=[(100.0, 100.0), (300.0, 100.0), (300.0, 250.0), (100.0, 250.0)],
        )

        detector = MotionDetector(
            MotionConfig(
                motion_threshold=150.0,
                roi_polygon=zone.points,
            )
        )
        blank = create_blank_frame()
        detector.update(blank, timestamp=0.0)

        # Die outside zone
        f_out = draw_test_die(blank, (500, 100), size=40)
        res_out = detector.update(f_out, timestamp=0.1)
        self.assertFalse(res_out.motion_detected)

        # Die inside zone
        f_in = draw_test_die(f_out, (200, 180), size=40)
        res_in = detector.update(f_in, timestamp=0.2)
        self.assertTrue(res_in.motion_detected)


if __name__ == "__main__":
    unittest.main()
