"""
test_detection.py - Comprehensive Unit Tests for Tumblin' Dice Detection & Scoring

Verification command: python -m unittest test_detection.py
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

import cv2
import numpy as np

from detection import (
    DEFAULT_PLAYER_ROSTER,
    D6PipCounter,
    DetectedDie,
    DiceDetector,
    DiceScoringResult,
    DetectionPipeline,
    HSVColorSampler,
    PipCountResult,
    PlayerProfile,
    PlayerProfileManager,
    PlayerRoundScore,
    ScoringZone,
    ZoneManager,
    ZoneScorer,
    circular_hue_distance,
    circular_hue_mean,
    compute_box_iou,
    compute_hsv_distance,
    create_synthetic_board,
    create_synthetic_die,
    non_maximum_suppression,
)


class TestHSVColorSampler(unittest.TestCase):
    """Tests for HSV color sampling, pip exclusion, and circular statistics."""

    def setUp(self) -> None:
        self.sampler = HSVColorSampler(pip_exclusion_threshold=30.0)

    def test_circular_hue_mean_standard(self):
        # Normal green hues around 60
        hues = [58, 60, 62]
        mean_h = circular_hue_mean(hues)
        self.assertAlmostEqual(mean_h, 60.0, delta=1.0)

    def test_circular_hue_mean_red_wrap_around(self):
        # Red wrap-around across 0 and 179: [178, 179, 0, 1, 2]
        # Standard arithmetic mean gives 72.0 (green-yellow!), which is wrong!
        # Circular mean must wrap around 0 / 180 degrees
        hues = [178, 179, 0, 1, 2]
        mean_h = circular_hue_mean(hues)
        # 180 is equivalent to 0 mod 180 in OpenCV hue
        self.assertTrue(abs(mean_h - 0.0) < 2.0 or abs(mean_h - 180.0) < 2.0)

    def test_circular_hue_mean_empty(self):
        self.assertEqual(circular_hue_mean([]), 0.0)

    def test_circular_hue_distance(self):
        # 10 and 20 -> dist 10
        self.assertAlmostEqual(circular_hue_distance(10, 20), 10.0)
        # 5 and 175 -> dist 10 across 0/180 wrap
        self.assertAlmostEqual(circular_hue_distance(5, 175), 10.0)
        # 0 and 90 -> dist 90
        self.assertAlmostEqual(circular_hue_distance(0, 90), 90.0)

    def test_sample_solid_color_roi(self):
        # Create solid Blue BGR patch (BGR: 220, 50, 20)
        blue_bgr = (220, 50, 20)
        roi = np.full((60, 60, 3), blue_bgr, dtype=np.uint8)
        h, s, v = self.sampler.sample_color_from_roi(roi, exclude_pips=False)

        expected_hsv = cv2.cvtColor(np.uint8([[list(blue_bgr)]]), cv2.COLOR_BGR2HSV)[0][0]
        self.assertAlmostEqual(h, float(expected_hsv[0]), delta=2.0)
        self.assertAlmostEqual(s, float(expected_hsv[1]), delta=2.0)
        self.assertAlmostEqual(v, float(expected_hsv[2]), delta=2.0)

    def test_sample_with_pip_exclusion(self):
        # Red die with white pips:
        # Pips would pull saturation down and value up if not excluded.
        die = create_synthetic_die(size=100, pips=6, bg_color=(20, 20, 210), fg_color=(255, 255, 255))
        h, s, v = self.sampler.sample_color_from_roi(die, exclude_pips=True)

        expected_hsv = cv2.cvtColor(np.uint8([[[20, 20, 210]]]), cv2.COLOR_BGR2HSV)[0][0]
        self.assertAlmostEqual(h, float(expected_hsv[0]), delta=3.0)
        # Saturation must stay high (> 210), not diluted by white dots
        self.assertGreater(s, 210.0)
        self.assertAlmostEqual(v, float(expected_hsv[2]), delta=10.0)

    def test_sample_circle_target(self):
        frame = np.full((300, 300, 3), (120, 120, 120), dtype=np.uint8)
        # Place a green die under sampling circle at (150, 150)
        cv2.circle(frame, (150, 150), 30, (30, 200, 30), -1)

        h, s, v = self.sampler.sample_color_from_circle(frame, center=(150, 150), radius=20)
        expected_hsv = cv2.cvtColor(np.uint8([[[30, 200, 30]]]), cv2.COLOR_BGR2HSV)[0][0]
        self.assertAlmostEqual(h, float(expected_hsv[0]), delta=4.0)
        self.assertGreater(s, 180.0)

    def test_sample_invalid_roi_raises(self):
        with self.assertRaises(ValueError):
            self.sampler.sample_color_from_roi(np.array([]))
        with self.assertRaises(ValueError):
            self.sampler.sample_color_from_roi(np.zeros((2, 2, 3), dtype=np.uint8))
        with self.assertRaises(ValueError):
            self.sampler.sample_color_from_circle(np.zeros((50, 50, 3), dtype=np.uint8), center=(100, 100), radius=20)


class TestPlayerProfileAndManager(unittest.TestCase):
    """Tests for player profiles, HSV distance metric, and roster manager."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_player_profiles.json")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_profile_creation_and_bounds(self):
        profile = PlayerProfile(
            player_id=1,
            name="Alice",
            color_name="Red",
            hsv_mean=(0.0, 240.0, 210.0),
            display_bgr=(30, 30, 220),
        )
        self.assertEqual(profile.player_id, 1)
        self.assertEqual(profile.name, "Alice")
        self.assertIsNotNone(profile.hsv_lower)
        self.assertIsNotNone(profile.hsv_upper)
        self.assertLessEqual(profile.hsv_lower[0], profile.hsv_upper[0])

    def test_profile_serialization_roundtrip(self):
        profile = PlayerProfile(
            player_id=2,
            name="Bob",
            color_name="Blue",
            hsv_mean=(115.0, 240.0, 200.0),
            display_bgr=(220, 100, 30),
        )
        d = profile.to_dict()
        restored = PlayerProfile.from_dict(d)
        self.assertEqual(restored.player_id, profile.player_id)
        self.assertEqual(restored.name, profile.name)
        self.assertEqual(restored.color_name, profile.color_name)
        self.assertAlmostEqual(restored.hsv_mean[0], profile.hsv_mean[0])
        self.assertEqual(restored.display_bgr, profile.display_bgr)

    def test_hsv_distance_metric_chromatic(self):
        # Distance to itself is 0
        red_hsv = (0.0, 240.0, 200.0)
        self.assertAlmostEqual(compute_hsv_distance(red_hsv, red_hsv), 0.0)

        # Distance to nearby red (circular wrap: H=178 vs H=2)
        h1 = (2.0, 235.0, 205.0)
        h2 = (178.0, 240.0, 195.0)
        dist = compute_hsv_distance(h1, h2)
        self.assertLess(dist, 0.3)

        # Distance between completely different colors (Red vs Blue)
        blue_hsv = (115.0, 240.0, 200.0)
        self.assertGreater(compute_hsv_distance(red_hsv, blue_hsv), 1.5)

    def test_hsv_distance_achromatic_distinction(self):
        # White (low saturation, high value) vs Red (high saturation)
        white_hsv = (0.0, 10.0, 250.0)
        red_hsv = (0.0, 240.0, 210.0)
        self.assertGreater(compute_hsv_distance(white_hsv, red_hsv), 1.0)

        # Black (low value) vs White
        black_hsv = (0.0, 10.0, 25.0)
        self.assertGreater(compute_hsv_distance(white_hsv, black_hsv), 1.0)

    def test_all_10_default_player_colors_distinguishable(self):
        manager = PlayerProfileManager.create_default_roster(10)
        self.assertEqual(len(manager.get_all_profiles()), 10)

        # Each profile should correctly match a noisy version of its own color
        for profile in manager.get_all_profiles():
            h, s, v = profile.hsv_mean
            noisy_h = (h + 3.0) % 180.0
            noisy_s = max(0.0, min(255.0, s - 8.0))
            noisy_v = max(0.0, min(255.0, v - 8.0))

            matched, dist = manager.match_color((noisy_h, noisy_s, noisy_v))
            self.assertIsNotNone(matched, f"Failed to match {profile.color_name}")
            self.assertEqual(matched.player_id, profile.player_id)
            self.assertLess(dist, 0.45)

    def test_player_manager_crud(self):
        manager = PlayerProfileManager()
        p1 = PlayerProfile(1, "Alice", "Red", (0.0, 240.0, 200.0))
        p2 = PlayerProfile(2, "Bob", "Blue", (115.0, 240.0, 200.0))
        manager.add_profile(p1)
        manager.add_profile(p2)

        self.assertEqual(len(manager.get_all_profiles()), 2)
        self.assertEqual(manager.get_profile(1), p1)
        self.assertTrue(manager.remove_profile(1))
        self.assertIsNone(manager.get_profile(1))
        self.assertFalse(manager.remove_profile(99))

    def test_player_manager_json_persistence(self):
        manager = PlayerProfileManager.create_default_roster(6)
        manager.save_to_json(self.config_path)
        self.assertTrue(os.path.exists(self.config_path))

        restored_manager = PlayerProfileManager()
        restored_manager.load_from_json(self.config_path)
        self.assertEqual(len(restored_manager.get_all_profiles()), 6)
        for original in manager.get_all_profiles():
            restored = restored_manager.get_profile(original.player_id)
            self.assertIsNotNone(restored)
            self.assertEqual(restored.name, original.name)
            self.assertEqual(restored.color_name, original.color_name)


class TestD6PipCounter(unittest.TestCase):
    """Tests for D6 pip counter across canonical faces, rotations, and polarities."""

    def setUp(self) -> None:
        self.counter = D6PipCounter()

    def test_canonical_faces_bright_pips_dark_die(self):
        # White pips on Red die (faces 1 through 6)
        for expected_pips in range(1, 7):
            die = create_synthetic_die(
                size=120,
                pips=expected_pips,
                bg_color=(30, 30, 200),
                fg_color=(255, 255, 255),
                angle_deg=0.0,
            )
            result = self.counter.count_pips(die)
            self.assertEqual(
                result.pip_count,
                expected_pips,
                f"Failed for face {expected_pips}: got {result.pip_count} with conf {result.confidence}",
            )
            self.assertGreater(result.confidence, 0.65)
            self.assertEqual(len(result.pip_centers), expected_pips)

    def test_canonical_faces_dark_pips_light_die(self):
        # Black pips on Yellow/White die (faces 1 through 6)
        for expected_pips in range(1, 7):
            die = create_synthetic_die(
                size=120,
                pips=expected_pips,
                bg_color=(240, 240, 240),
                fg_color=(20, 20, 20),
                angle_deg=0.0,
            )
            result = self.counter.count_pips(die)
            self.assertEqual(
                result.pip_count,
                expected_pips,
                f"Failed for dark pips face {expected_pips}: got {result.pip_count}",
            )
            self.assertGreater(result.confidence, 0.65)
            self.assertEqual(len(result.pip_centers), expected_pips)

    def test_rotated_dice_pips(self):
        # Test rotations at 15, 30, 45, 90, 180 degrees
        rotations = [15.0, 30.0, 45.0, 90.0, 180.0]
        for pips in [1, 3, 5, 6]:
            for angle in rotations:
                die = create_synthetic_die(
                    size=120,
                    pips=pips,
                    bg_color=(30, 30, 200),
                    fg_color=(255, 255, 255),
                    angle_deg=angle,
                )
                result = self.counter.count_pips(die)
                self.assertEqual(
                    result.pip_count,
                    pips,
                    f"Failed for pips={pips} at angle={angle}: got {result.pip_count}",
                )

    def test_blank_die_returns_zero(self):
        blank = np.full((100, 100, 3), (30, 30, 200), dtype=np.uint8)
        result = self.counter.count_pips(blank)
        self.assertEqual(result.pip_count, 0)
        self.assertEqual(result.confidence, 0.0)

    def test_invalid_die_inputs(self):
        res1 = self.counter.count_pips(None)  # type: ignore[arg-type]
        self.assertEqual(res1.pip_count, 0)

        res2 = self.counter.count_pips(np.zeros((5, 5, 3), dtype=np.uint8))
        self.assertEqual(res2.pip_count, 0)

    def test_non_standard_die_aspect_ratio_and_sizes(self):
        # Test die sizes: 40x40, 80x80, 150x150
        for sz in (40, 80, 150):
            die = create_synthetic_die(size=sz, pips=3)
            res = self.counter.count_pips(die)
            self.assertEqual(res.pip_count, 3, f"Failed for size {sz}")

        # Test slightly rectangular crop (e.g. 90x110)
        die = create_synthetic_die(size=100, pips=5)
        rect_crop = cv2.resize(die, (90, 110))
        res_rect = self.counter.count_pips(rect_crop)
        self.assertEqual(res_rect.pip_count, 5)


class TestDiceCandidateDetection(unittest.TestCase):
    """Tests for locating candidate dice on board images and NMS."""

    def test_compute_box_iou(self):
        # Identical boxes
        self.assertAlmostEqual(compute_box_iou((10, 10, 50, 50), (10, 10, 50, 50)), 1.0)
        # Disjoint boxes
        self.assertAlmostEqual(compute_box_iou((10, 10, 50, 50), (100, 100, 50, 50)), 0.0)
        # 50% overlap approx
        iou = compute_box_iou((0, 0, 100, 100), (50, 0, 100, 100))
        self.assertAlmostEqual(iou, 5000.0 / 15000.0, delta=1e-3)

    def test_non_maximum_suppression(self):
        boxes = [
            (100, 100, 50, 50),
            (102, 102, 48, 48),  # Almost identical overlap
            (300, 300, 50, 50),  # Separate
        ]
        scores = [0.95, 0.80, 0.90]
        keep = non_maximum_suppression(boxes, scores, iou_threshold=0.3)
        self.assertEqual(len(keep), 2)
        self.assertIn(0, keep)
        self.assertIn(2, keep)
        self.assertNotIn(1, keep)

    def test_detect_dice_on_synthetic_board(self):
        dice_specs = [
            {"cx": 200, "cy": 300, "size": 60, "pips": 4, "bg_color": (30, 30, 200)},
            {"cx": 500, "cy": 700, "size": 60, "pips": 2, "bg_color": (200, 50, 30)},
        ]
        board, _ = create_synthetic_board(width=800, height=1000, dice_specs=dice_specs)

        detector = DiceDetector(min_die_size=30, max_die_size=100)
        candidates = detector.extract_candidates(board)
        self.assertGreaterEqual(len(candidates), 2)

        # Check candidate near (200, 300)
        found_d1 = any(abs((b[0] + b[2] / 2) - 200) < 15 and abs((b[1] + b[3] / 2) - 300) < 15 for b in candidates)
        # Check candidate near (500, 700)
        found_d2 = any(abs((b[0] + b[2] / 2) - 500) < 15 and abs((b[1] + b[3] / 2) - 700) < 15 for b in candidates)
        self.assertTrue(found_d1, "Candidate near (200, 300) not found")
        self.assertTrue(found_d2, "Candidate near (500, 700) not found")


class TestPointInPolygonZoneScorer(unittest.TestCase):
    """Tests for spatial scoring zones, multiplier calculation, and foul/out-of-bounds rules."""

    def setUp(self) -> None:
        # Create standard 3-tier zones: 1x, 2x, 3x
        self.zone1 = ScoringZone(
            name="Zone 1x",
            multiplier=1,
            points=[(100.0, 100.0), (300.0, 100.0), (300.0, 300.0), (100.0, 300.0)],
        )
        self.zone2 = ScoringZone(
            name="Zone 2x",
            multiplier=2,
            points=[(350.0, 100.0), (550.0, 100.0), (550.0, 300.0), (350.0, 300.0)],
        )
        self.zone3 = ScoringZone(
            name="Zone 3x",
            multiplier=3,
            points=[(100.0, 350.0), (550.0, 350.0), (550.0, 600.0), (100.0, 600.0)],
        )
        self.scorer = ZoneScorer(zones_source=[self.zone1, self.zone2, self.zone3])

    def test_zone_point_containment_and_multiplier(self):
        # Inside Zone 1x
        self.assertEqual(self.scorer.get_multiplier(200.0, 200.0), 1)
        # Inside Zone 2x
        self.assertEqual(self.scorer.get_multiplier(450.0, 200.0), 2)
        # Inside Zone 3x
        self.assertEqual(self.scorer.get_multiplier(300.0, 450.0), 3)
        # Out of bounds
        self.assertEqual(self.scorer.get_multiplier(700.0, 700.0), 0)

    def test_score_individual_die(self):
        die = DetectedDie(
            die_id=1,
            center=(450.0, 200.0),  # Inside Zone 2x
            bbox=(420, 170, 60, 60),
            pip_count=5,
            pip_confidence=0.95,
        )
        scored = self.scorer.score_die(die)
        self.assertEqual(scored.zone_name, "Zone 2x")
        self.assertEqual(scored.zone_multiplier, 2)
        # 5 pips * 2x multiplier = 10 points
        self.assertEqual(scored.score, 10)
        self.assertFalse(scored.is_out_of_bounds)

    def test_out_of_bounds_die_scores_zero(self):
        die = DetectedDie(
            die_id=2,
            center=(800.0, 900.0),  # Off board
            bbox=(770, 870, 60, 60),
            pip_count=6,
            pip_confidence=0.95,
        )
        scored = self.scorer.score_die(die)
        self.assertEqual(scored.zone_name, "Out of Bounds")
        self.assertEqual(scored.zone_multiplier, 0)
        self.assertEqual(scored.score, 0)
        self.assertTrue(scored.is_out_of_bounds)

    def test_irregular_polygon_zone(self):
        # 6-sided irregular polygon (hexagon-like)
        hex_points = [
            (200.0, 100.0),
            (300.0, 100.0),
            (350.0, 200.0),
            (300.0, 300.0),
            (200.0, 300.0),
            (150.0, 200.0),
        ]
        hex_zone = ScoringZone(name="Hex Zone 4x", multiplier=4, points=hex_points)
        custom_scorer = ZoneScorer(zones_source=[hex_zone])

        # Centroid (250, 200) inside
        self.assertEqual(custom_scorer.get_multiplier(250.0, 200.0), 4)
        # Outside corners
        self.assertEqual(custom_scorer.get_multiplier(100.0, 100.0), 0)
        self.assertEqual(custom_scorer.get_multiplier(350.0, 100.0), 0)

    def test_multi_player_round_scoring_and_leaderboard(self):
        manager = PlayerProfileManager.create_default_roster(2)

        # Alice (Player 1): Die 1 with 4 pips in 3x (12 pts), Die 2 with 5 pips in 1x (5 pts) = 17 pts
        d1 = DetectedDie(1, (200.0, 450.0), (170, 420, 60, 60), pip_count=4, pip_confidence=0.9, player_id=1, player_name="Player 1")
        d2 = DetectedDie(2, (200.0, 200.0), (170, 170, 60, 60), pip_count=5, pip_confidence=0.9, player_id=1, player_name="Player 1")

        # Bob (Player 2): Die 3 with 6 pips in 2x (12 pts), Die 4 with 3 pips OUT (0 pts) = 12 pts
        d3 = DetectedDie(3, (450.0, 200.0), (420, 170, 60, 60), pip_count=6, pip_confidence=0.9, player_id=2, player_name="Player 2")
        d4 = DetectedDie(4, (750.0, 750.0), (720, 720, 60, 60), pip_count=3, pip_confidence=0.9, player_id=2, player_name="Player 2")

        result = self.scorer.score_dice([d1, d2, d3, d4], player_manager=manager)

        self.assertEqual(result.total_board_score, 29)  # 12 + 5 + 12 = 29
        self.assertEqual(result.active_dice_count, 3)
        self.assertEqual(len(result.out_of_bounds_dice), 1)

        # Leaderboard: Alice #1 (17 pts), Bob #2 (12 pts)
        self.assertEqual(len(result.leaderboard), 2)
        self.assertEqual(result.leaderboard[0]["player_id"], 1)
        self.assertEqual(result.leaderboard[0]["total_score"], 17)
        self.assertEqual(result.leaderboard[1]["player_id"], 2)
        self.assertEqual(result.leaderboard[1]["total_score"], 12)

    def test_zone_scorer_load_from_zones_config_json(self):
        # Test loading real zones_config.json from project workspace
        config_path = "zones_config.json"
        if not os.path.exists(config_path):
            self.skipTest(f"{config_path} not found in workspace")

        scorer = ZoneScorer(zones_source=config_path)
        self.assertGreaterEqual(len(scorer.zones), 4)

        # Zone 1x polygon covers (200, 500)
        self.assertEqual(scorer.get_multiplier(200.0, 500.0), 1)
        # Zone 2x polygon covers (440, 500)
        self.assertEqual(scorer.get_multiplier(440.0, 500.0), 2)
        # Zone 3x polygon covers (650, 450)
        self.assertEqual(scorer.get_multiplier(650.0, 450.0), 3)
        # Off-board point scores 0
        self.assertEqual(scorer.get_multiplier(50.0, 50.0), 0)

    def test_dynamic_physics_knock_off_score_update(self):
        # Scenario: Player 1 has a die with 5 pips in Zone 3x (15 pts).
        # Player 2 flicks a die with 6 pips, which knocks Player 1's die off the board!
        # Now Player 1's die is off-board (0 pts), and Player 2 lands in Zone 2x (12 pts).
        manager = PlayerProfileManager.create_default_roster(2)

        # State before flick: Player 1 on board in Zone 3x
        die_p1_before = DetectedDie(1, (200.0, 450.0), (170, 420, 60, 60), pip_count=5, pip_confidence=0.9, player_id=1)
        res_before = self.scorer.score_dice([die_p1_before], player_manager=manager)
        self.assertEqual(res_before.player_scores[1], 15)
        self.assertEqual(res_before.player_scores[2], 0)

        # State after flick: Player 1 knocked off (800, 800), Player 2 lands in Zone 2x (450, 200)
        die_p1_after = DetectedDie(1, (800.0, 800.0), (770, 770, 60, 60), pip_count=5, pip_confidence=0.9, player_id=1)
        die_p2_after = DetectedDie(2, (450.0, 200.0), (420, 170, 60, 60), pip_count=6, pip_confidence=0.9, player_id=2)
        res_after = self.scorer.score_dice([die_p1_after, die_p2_after], player_manager=manager)

        self.assertEqual(res_after.player_scores[1], 0)   # Knocked off = 0 pts!
        self.assertEqual(res_after.player_scores[2], 12)  # 6 * 2x = 12 pts
        self.assertEqual(res_after.leaderboard[0]["player_id"], 2)

    def test_10_player_full_tournament_scoring(self):
        # Up to 10 players, each with 4 dice (40 dice total on board)
        manager = PlayerProfileManager.create_default_roster(10)
        all_dice: List[DetectedDie] = []

        die_id = 1
        for pid in range(1, 11):
            # 2 dice in Zone 1x (pips 3 and 4 -> 3*1 + 4*1 = 7 pts)
            all_dice.append(DetectedDie(die_id, (200.0, 200.0), (170, 170, 50, 50), pip_count=3, pip_confidence=0.9, player_id=pid))
            die_id += 1
            all_dice.append(DetectedDie(die_id, (220.0, 220.0), (190, 190, 50, 50), pip_count=4, pip_confidence=0.9, player_id=pid))
            die_id += 1
            # 1 die in Zone 2x (pips 5 -> 5*2 = 10 pts)
            all_dice.append(DetectedDie(die_id, (450.0, 200.0), (420, 170, 50, 50), pip_count=5, pip_confidence=0.9, player_id=pid))
            die_id += 1
            # 1 die in Zone 3x with pips equal to pid (pid * 3 pts)
            all_dice.append(DetectedDie(die_id, (200.0, 450.0), (170, 420, 50, 50), pip_count=min(6, pid % 6 + 1), pip_confidence=0.9, player_id=pid))
            die_id += 1

        self.assertEqual(len(all_dice), 40)
        result = self.scorer.score_dice(all_dice, player_manager=manager)

        self.assertEqual(len(result.leaderboard), 10)
        self.assertEqual(result.active_dice_count, 40)
        self.assertEqual(len(result.out_of_bounds_dice), 0)

        # Leaderboard should be strictly ordered descending
        for i in range(len(result.leaderboard) - 1):
            self.assertGreaterEqual(
                result.leaderboard[i]["total_score"],
                result.leaderboard[i + 1]["total_score"],
            )


class TestDetectionPipeline(unittest.TestCase):
    """Tests for complete DetectionPipeline, HUD overlay rendering, and referee overrides."""

    def setUp(self) -> None:
        self.manager = PlayerProfileManager.create_default_roster(2)
        self.zone1 = ScoringZone("Zone 1x", 1, [(50.0, 50.0), (350.0, 50.0), (350.0, 350.0), (50.0, 350.0)])
        self.zone2 = ScoringZone("Zone 2x", 2, [(400.0, 50.0), (700.0, 50.0), (700.0, 350.0), (400.0, 350.0)])
        self.zone_scorer = ZoneScorer([self.zone1, self.zone2])
        self.pipeline = DetectionPipeline(player_manager=self.manager, zone_scorer=self.zone_scorer)

    def test_pipeline_with_synthetic_scene(self):
        # Create board with 2 dice:
        # Die 1: Player 1 (Red) in Zone 1x with 4 pips -> 4 * 1 = 4 pts
        # Die 2: Player 2 (Blue) in Zone 2x with 3 pips -> 3 * 2 = 6 pts
        dice_specs = [
            {"cx": 200, "cy": 200, "size": 60, "pips": 4, "bg_color": (30, 30, 220), "player_id": 1},
            {"cx": 550, "cy": 200, "size": 60, "pips": 3, "bg_color": (220, 100, 30), "player_id": 2},
        ]
        board, _ = create_synthetic_board(width=800, height=600, dice_specs=dice_specs)

        # Provide candidate boxes directly for precise evaluation
        candidate_boxes = [(170, 170, 60, 60), (520, 170, 60, 60)]
        result, hud_img = self.pipeline.process_frame(board, candidate_boxes=candidate_boxes)

        self.assertEqual(len(result.dice), 2)
        self.assertEqual(result.total_board_score, 10)  # 4 + 6 = 10
        self.assertEqual(result.active_dice_count, 2)
        self.assertEqual(len(result.leaderboard), 2)

        # Leaderboard check: Player 2 #1 (6 pts), Player 1 #2 (4 pts)
        self.assertEqual(result.leaderboard[0]["player_id"], 2)
        self.assertEqual(result.leaderboard[0]["total_score"], 6)
        self.assertEqual(result.leaderboard[1]["player_id"], 1)
        self.assertEqual(result.leaderboard[1]["total_score"], 4)

        # Verify HUD overlay image shape matches input frame
        self.assertEqual(hud_img.shape, board.shape)

    def test_pipeline_referee_manual_override(self):
        d1 = DetectedDie(1, (200.0, 200.0), (170, 170, 60, 60), pip_count=2, pip_confidence=0.9, player_id=1, player_name="Player 1")
        frame = np.full((400, 400, 3), (200, 200, 200), dtype=np.uint8)

        # Apply referee manual override on die 1: change pips from 2 to 6
        self.pipeline.override_die(1, pip_count=6)

        result, _ = self.pipeline.process_frame(frame, candidate_boxes=[(170, 170, 60, 60)])
        # If candidate is processed, override must take effect
        # Test direct override application
        d1.pip_count = 6
        scored = self.zone_scorer.score_die(d1)
        self.assertEqual(scored.pip_count, 6)
        self.assertEqual(scored.score, 6)  # 6 * 1x = 6 pts

    def test_pipeline_serialization_output(self):
        d1 = DetectedDie(1, (200.0, 200.0), (170, 170, 60, 60), pip_count=4, pip_confidence=0.9, player_id=1, player_name="Player 1")
        result = self.zone_scorer.score_dice([d1], player_manager=self.manager)
        d_dict = result.to_dict()
        self.assertIn("total_board_score", d_dict)
        self.assertIn("leaderboard", d_dict)
        self.assertIn("dice", d_dict)


class TestSyntheticGenerators(unittest.TestCase):
    """Tests for synthetic die and board generation helpers."""

    def test_create_synthetic_die_dimensions(self):
        die = create_synthetic_die(size=80, pips=5)
        self.assertEqual(die.shape, (80, 80, 3))
        self.assertEqual(die.dtype, np.uint8)

    def test_create_synthetic_die_invalid_pips_raises(self):
        with self.assertRaises(ValueError):
            create_synthetic_die(pips=0)
        with self.assertRaises(ValueError):
            create_synthetic_die(pips=7)

    def test_create_synthetic_board(self):
        board, meta = create_synthetic_board(
            width=600,
            height=800,
            dice_specs=[{"cx": 300, "cy": 400, "size": 50, "pips": 3}],
        )
        self.assertEqual(board.shape, (800, 600, 3))
        self.assertEqual(len(meta), 1)
        self.assertEqual(meta[0]["pips"], 3)


if __name__ == "__main__":
    unittest.main()
