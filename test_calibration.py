"""
test_calibration.py - Comprehensive Unit Tests for Tumblin' Dice Calibration & Zone Editor

Verification command: python -m unittest test_calibration.py
"""

import json
import os
import shutil
import tempfile
import unittest

import cv2
import numpy as np

from calibration import (
    DEFAULT_BOARD_HEIGHT,
    DEFAULT_BOARD_WIDTH,
    DEFAULT_ZONE_COLORS,
    VALID_MULTIPLIERS,
    ArucoCalibrator,
    CalibrationPipeline,
    InteractiveZoneEditor,
    ScoringZone,
    ZoneManager,
    generate_aruco_calibration_sheet,
    order_points_clockwise,
)


class TestPointOrdering(unittest.TestCase):
    """Tests for clockwise point ordering utility."""

    def test_order_axis_aligned_rectangle(self):
        # Scrambled coordinates: TR, BL, BR, TL
        scrambled = [[800.0, 0.0], [0.0, 1200.0], [800.0, 1200.0], [0.0, 0.0]]
        ordered = order_points_clockwise(scrambled)
        # Expected: TL, TR, BR, BL
        np.testing.assert_allclose(ordered[0], [0.0, 0.0], atol=1e-3)
        np.testing.assert_allclose(ordered[1], [800.0, 0.0], atol=1e-3)
        np.testing.assert_allclose(ordered[2], [800.0, 1200.0], atol=1e-3)
        np.testing.assert_allclose(ordered[3], [0.0, 1200.0], atol=1e-3)

    def test_order_rotated_quadrilateral(self):
        # Quad rotated by 30 degrees around (500, 500)
        theta = np.radians(30)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        corners = np.array([[200, 200], [800, 200], [800, 700], [200, 700]], dtype=np.float32)
        rotated = (corners - [500, 450]) @ R.T + [500, 450]

        # Shuffle points
        shuffled = rotated[[2, 0, 3, 1]]
        ordered = order_points_clockwise(shuffled)
        self.assertEqual(ordered.shape, (4, 2))
        # Top-left anchor should have smallest x+y
        self.assertLess(ordered[0][0] + ordered[0][1], ordered[2][0] + ordered[2][1])

    def test_order_invalid_shapes(self):
        with self.assertRaises(ValueError):
            order_points_clockwise([[0, 0], [1, 1], [2, 2]])  # Only 3 points
        with self.assertRaises(ValueError):
            order_points_clockwise(np.zeros((5, 2)))  # 5 points
        with self.assertRaises(ValueError):
            # Duplicate / degenerate points
            order_points_clockwise([[0, 0], [0, 0], [10, 10], [10, 0]])


class TestScoringZone(unittest.TestCase):
    """Tests for ScoringZone polygon model, point containment, and multipliers."""

    def test_valid_zone_creation(self):
        pts = [(100.0, 100.0), (300.0, 100.0), (300.0, 300.0), (100.0, 300.0)]
        for mult in VALID_MULTIPLIERS:
            zone = ScoringZone(name=f"Zone {mult}x", multiplier=mult, points=pts)
            self.assertEqual(zone.multiplier, mult)
            self.assertEqual(zone.name, f"Zone {mult}x")
            self.assertEqual(len(zone.points), 4)
            self.assertEqual(zone.color, DEFAULT_ZONE_COLORS[mult])

    def test_invalid_multiplier_raises(self):
        pts = [(100, 100), (200, 100), (200, 200)]
        with self.assertRaises(ValueError):
            ScoringZone("Too High", 7, pts)
        with self.assertRaises(ValueError):
            ScoringZone("Negative", -1, pts)
        with self.assertRaises(TypeError):
            ScoringZone("String Mult", "3", pts)  # type: ignore

    def test_insufficient_points_raises(self):
        with self.assertRaises(ValueError):
            ScoringZone("Line", 1, [(0, 0), (100, 100)])
        with self.assertRaises(ValueError):
            ScoringZone("Empty", 1, [])

    def test_contains_point_convex(self):
        # 200x200 square at (100, 100)
        pts = [(100.0, 100.0), (300.0, 100.0), (300.0, 300.0), (100.0, 300.0)]
        zone = ScoringZone("Zone 2x", 2, pts)

        self.assertTrue(zone.contains_point(200, 200))   # Inside center
        self.assertTrue(zone.contains_point(100, 100))   # On corner vertex
        self.assertTrue(zone.contains_point(200, 100))   # On top edge
        self.assertFalse(zone.contains_point(50, 50))    # Outside top-left
        self.assertFalse(zone.contains_point(350, 200))  # Outside right
        self.assertFalse(zone.contains_point(200, 350))  # Outside bottom

    def test_contains_point_irregular_polygon(self):
        # L-shaped polygon (6 vertices) with a cutout notch on top-right:
        # (100,100) -> (200,100) -> (200,200) -> (300,200) -> (300,300) -> (100,300)
        pts = [
            (100.0, 100.0),
            (200.0, 100.0),
            (200.0, 200.0),
            (300.0, 200.0),
            (300.0, 300.0),
            (100.0, 300.0),
        ]
        zone = ScoringZone("L-Zone", 3, pts)

        self.assertTrue(zone.contains_point(150, 150))   # Inside upper-left stalk
        self.assertTrue(zone.contains_point(250, 250))   # Inside lower-right foot
        self.assertFalse(zone.contains_point(250, 150))  # In the cutout notch (outside!)
        self.assertFalse(zone.contains_point(50, 50))    # Outside

    def test_distance_to_point(self):
        pts = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        zone = ScoringZone("Square", 1, pts)
        self.assertGreater(zone.distance_to_point(50, 50), 0)   # Positive inside
        self.assertLess(zone.distance_to_point(150, 50), 0)     # Negative outside
        self.assertAlmostEqual(zone.distance_to_point(0, 50), 0.0, delta=1e-3)  # Zero on edge

    def test_centroid_and_area(self):
        pts = [(0.0, 0.0), (200.0, 0.0), (200.0, 100.0), (0.0, 100.0)]
        zone = ScoringZone("Rect", 4, pts)
        cx, cy = zone.centroid
        self.assertAlmostEqual(cx, 100.0, delta=1.0)
        self.assertAlmostEqual(cy, 50.0, delta=1.0)
        self.assertAlmostEqual(zone.area, 20000.0, delta=1.0)

    def test_serialization_roundtrip(self):
        pts = [(50.0, 50.0), (150.0, 50.0), (100.0, 120.0)]
        custom_color = (12, 34, 56)
        zone = ScoringZone("Triangle 5x", 5, pts, color=custom_color)
        d = zone.to_dict()

        self.assertEqual(d["name"], "Triangle 5x")
        self.assertEqual(d["multiplier"], 5)
        self.assertEqual(d["color"], [12, 34, 56])

        restored = ScoringZone.from_dict(d)
        self.assertEqual(restored.name, zone.name)
        self.assertEqual(restored.multiplier, zone.multiplier)
        self.assertEqual(restored.color, custom_color)
        np.testing.assert_allclose(restored.points, zone.points)


class TestZoneManager(unittest.TestCase):
    """Tests for ZoneManager spatial queries, overlay rendering, and JSON persistence."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_zones.json")
        self.manager = ZoneManager(board_width=800, board_height=1200)

        # Setup standard tiered zones 1x through 6x
        for m in VALID_MULTIPLIERS:
            y1 = (m - 1) * 180 + 50
            y2 = y1 + 160
            pts = [(50.0, float(y1)), (750.0, float(y1)), (750.0, float(y2)), (50.0, float(y2))]
            self.manager.add_zone(ScoringZone(name=f"Tier {m}x", multiplier=m, points=pts))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_add_and_query_multipliers(self):
        self.assertEqual(len(self.manager.zones), 6)
        # Check center of each tier
        for m in VALID_MULTIPLIERS:
            test_y = (m - 1) * 180 + 50 + 80
            mult = self.manager.get_multiplier(400, test_y)
            self.assertEqual(mult, m, f"Failed for multiplier {m}")

    def test_out_of_bounds_returns_zero(self):
        # Coordinate outside any zone
        self.assertEqual(self.manager.get_multiplier(10, 10), 0)
        self.assertIsNone(self.manager.find_zone(10, 10))
        self.assertEqual(self.manager.get_multiplier(400, 1190), 0)

    def test_remove_and_clear_zones(self):
        removed = self.manager.remove_zone(0)
        self.assertEqual(removed.multiplier, 1)
        self.assertEqual(len(self.manager.zones), 5)

        self.assertTrue(self.manager.remove_zone_by_name("Tier 6x"))
        self.assertEqual(len(self.manager.zones), 4)

        self.manager.clear_zones()
        self.assertEqual(len(self.manager.zones), 0)

    def test_render_overlay(self):
        canvas = np.full((1200, 800, 3), 200, dtype=np.uint8)
        rendered = self.manager.render_overlay(canvas, alpha=0.35, show_labels=True)
        self.assertEqual(rendered.shape, canvas.shape)
        self.assertEqual(rendered.dtype, np.uint8)
        # Check that canvas was modified by overlay drawing
        self.assertFalse(np.array_equal(rendered, canvas))

    def test_json_save_and_load(self):
        self.manager.save_to_json(self.config_path)
        self.assertTrue(os.path.exists(self.config_path))

        new_manager = ZoneManager()
        new_manager.load_from_json(self.config_path)

        self.assertEqual(len(new_manager.zones), 6)
        self.assertEqual(new_manager.board_width, 800)
        self.assertEqual(new_manager.board_height, 1200)

        for orig, loaded in zip(self.manager.zones, new_manager.zones):
            self.assertEqual(orig.name, loaded.name)
            self.assertEqual(orig.multiplier, loaded.multiplier)
            np.testing.assert_allclose(orig.points, loaded.points)

    def test_load_raw_list_json_format(self):
        raw_list = [
            {"name": "Z1", "multiplier": 1, "points": [[0, 0], [10, 0], [10, 10], [0, 10]]},
            {"name": "Z4", "multiplier": 4, "points": [[20, 20], [30, 20], [30, 30], [20, 30]]},
        ]
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(raw_list, f)

        new_manager = ZoneManager()
        new_manager.load_from_json(self.config_path)
        self.assertEqual(len(new_manager.zones), 2)
        self.assertEqual(new_manager.get_zone(0).multiplier, 1)
        self.assertEqual(new_manager.get_zone(1).multiplier, 4)

    def test_load_nonexistent_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.manager.load_from_json("nonexistent_file_xyz123.json")

    def test_invalid_zone_type_raises(self):
        with self.assertRaises(TypeError):
            self.manager.add_zone("not_a_zone")  # type: ignore

    def test_remove_zone_out_of_bounds_raises(self):
        with self.assertRaises(IndexError):
            self.manager.remove_zone(99)

    def test_corrupted_json_raises(self):
        corrupted_path = os.path.join(self.temp_dir, "corrupt.json")
        with open(corrupted_path, "w", encoding="utf-8") as f:
            f.write("{invalid json content!}")
        with self.assertRaises(json.JSONDecodeError):
            self.manager.load_from_json(corrupted_path)


class TestArucoCalibrator(unittest.TestCase):
    """Tests for ArUco 4-corner detection, homography matrix calculation, and perspective warping."""

    def setUp(self):
        self.width = 800
        self.height = 1200
        self.sheet = generate_aruco_calibration_sheet(
            width=self.width, height=self.height, marker_size=100, margin=50
        )
        self.calibrator = ArucoCalibrator(
            board_width=self.width,
            board_height=self.height,
            marker_inset_x=100.0,  # 50 margin + 50 half-marker
            marker_inset_y=100.0,
        )

    def test_detect_markers_on_sheet(self):
        corners, centers = self.calibrator.detect_markers(self.sheet)
        self.assertEqual(len(corners), 4)
        self.assertEqual(len(centers), 4)
        for mid in [0, 1, 2, 3]:
            self.assertIn(mid, centers)

        # TL marker (0) center should be near (100, 100)
        self.assertAlmostEqual(centers[0][0], 100.0, delta=2.0)
        self.assertAlmostEqual(centers[0][1], 100.0, delta=2.0)

        # TR marker (1) center near (700, 100)
        self.assertAlmostEqual(centers[1][0], 700.0, delta=2.0)
        self.assertAlmostEqual(centers[1][1], 100.0, delta=2.0)

        # BR marker (2) center near (700, 1100)
        self.assertAlmostEqual(centers[2][0], 700.0, delta=2.0)
        self.assertAlmostEqual(centers[2][1], 1100.0, delta=2.0)

        # BL marker (3) center near (100, 1100)
        self.assertAlmostEqual(centers[3][0], 100.0, delta=2.0)
        self.assertAlmostEqual(centers[3][1], 1100.0, delta=2.0)

    def test_calibration_from_sheet(self):
        success = self.calibrator.calibrate_from_frame(self.sheet)
        self.assertTrue(success)
        self.assertIsNotNone(self.calibrator.homography_matrix)
        self.assertIsNotNone(self.calibrator.inv_homography_matrix)
        self.assertEqual(self.calibrator.homography_matrix.shape, (3, 3))

    def test_calibration_failure_insufficient_markers(self):
        blank = np.full((1200, 800, 3), 255, dtype=np.uint8)
        success = self.calibrator.calibrate_from_frame(blank)
        self.assertFalse(success)
        self.assertIsNone(self.calibrator.homography_matrix)

    def test_perspective_warp_and_inversion(self):
        # Simulate skewed camera perspective by warping sheet onto 1280x960 camera view
        src_rect = np.float32([[0, 0], [self.width, 0], [self.width, self.height], [0, self.height]])
        dst_quad = np.float32([[200, 150], [950, 100], [1050, 850], [150, 800]])
        H_tilt = cv2.getPerspectiveTransform(src_rect, dst_quad)
        camera_frame = cv2.warpPerspective(self.sheet, H_tilt, (1280, 960))

        # Calibrate on this tilted camera frame
        success = self.calibrator.calibrate_from_frame(camera_frame)
        self.assertTrue(success)

        # Warp camera frame to rectified board plane
        warped = self.calibrator.warp_frame(camera_frame)
        self.assertEqual(warped.shape, (self.height, self.width, 3))

        # Test bidirectional point transformation
        # Board center (400, 600) -> camera coords -> recovered board coords
        pt_board = np.float32([[[400.0, 600.0]]])
        pt_cam = cv2.perspectiveTransform(pt_board, H_tilt)
        cam_x, cam_y = float(pt_cam[0][0][0]), float(pt_cam[0][0][1])

        bx, by = self.calibrator.transform_camera_to_board(cam_x, cam_y)
        # Recovered point should match (400, 600) within 1.0 pixel
        self.assertAlmostEqual(bx, 400.0, delta=1.0)
        self.assertAlmostEqual(by, 600.0, delta=1.0)

        # Check inverse board -> camera transform
        back_cam_x, back_cam_y = self.calibrator.transform_board_to_camera(bx, by)
        self.assertAlmostEqual(back_cam_x, cam_x, delta=1.0)
        self.assertAlmostEqual(back_cam_y, cam_y, delta=1.0)

    def test_vectorized_transform_points(self):
        self.calibrator.calibrate_from_frame(self.sheet)
        test_pts = np.array([[100, 100], [400, 600], [700, 1100]], dtype=np.float32)
        transformed = self.calibrator.transform_points(test_pts, to_board=True)
        self.assertEqual(transformed.shape, (3, 2))
        recovered = self.calibrator.transform_points(transformed, to_board=False)
        np.testing.assert_allclose(test_pts, recovered, atol=1e-1)

    def test_unwarp_frame(self):
        self.calibrator.calibrate_from_frame(self.sheet)
        warped = self.calibrator.warp_frame(self.sheet)
        unwarped = self.calibrator.unwarp_frame(warped, (self.height, self.width))
        self.assertEqual(unwarped.shape, self.sheet.shape)

    def test_warp_without_calibration_raises(self):
        uncalibrated = ArucoCalibrator()
        with self.assertRaises(RuntimeError):
            uncalibrated.warp_frame(self.sheet)

    def test_geometric_fallback_with_arbitrary_ids(self):
        # Sheet created with non-standard marker IDs: 10, 20, 30, 40
        sheet_custom = generate_aruco_calibration_sheet(
            width=self.width, height=self.height, marker_ids=[10, 20, 30, 40]
        )
        # Calibrator expects [0, 1, 2, 3] by default
        calibrator = ArucoCalibrator(board_width=self.width, board_height=self.height)
        success = calibrator.calibrate_from_frame(sheet_custom)
        self.assertTrue(success)
        self.assertIsNotNone(calibrator.homography_matrix)

    def test_manual_calibration_from_points(self):
        src = [[100, 100], [700, 100], [700, 1100], [100, 1100]]
        dst = [[0, 0], [800, 0], [800, 1200], [0, 1200]]
        calibrator = ArucoCalibrator(board_width=800, board_height=1200)
        success = calibrator.calibrate_from_points(src, dst)
        self.assertTrue(success)
        self.assertIsNotNone(calibrator.homography_matrix)

        # Check point (100, 100) maps to (0, 0)
        bx, by = calibrator.transform_camera_to_board(100, 100)
        self.assertAlmostEqual(bx, 0.0, delta=1e-2)
        self.assertAlmostEqual(by, 0.0, delta=1e-2)

    def test_calibrator_serialization(self):
        self.calibrator.calibrate_from_frame(self.sheet)
        d = self.calibrator.to_dict()
        self.assertIn("homography_matrix", d)
        self.assertEqual(d["board_width"], 800)

        restored = ArucoCalibrator()
        restored.from_dict(d)
        self.assertIsNotNone(restored.homography_matrix)
        np.testing.assert_allclose(restored.homography_matrix, self.calibrator.homography_matrix)


class TestInteractiveZoneEditor(unittest.TestCase):
    """Tests for InteractiveZoneEditor mouse events, hotkeys, and headless operation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "editor_zones.json")
        self.manager = ZoneManager(800, 1200)
        self.editor = InteractiveZoneEditor(
            zone_manager=self.manager,
            config_path=self.config_path,
            board_width=800,
            board_height=1200,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initial_state(self):
        self.assertEqual(self.editor.active_multiplier, 1)
        self.assertEqual(len(self.editor.current_points), 0)
        self.assertEqual(len(self.manager.zones), 0)

    def test_set_multiplier_keys(self):
        for m in VALID_MULTIPLIERS:
            handled = self.editor.handle_key(str(m))
            self.assertTrue(handled)
            self.assertEqual(self.editor.active_multiplier, m)

        with self.assertRaises(ValueError):
            self.editor.set_multiplier(7)
        with self.assertRaises(ValueError):
            self.editor.set_multiplier(0)

    def test_mouse_draw_and_finish_zone(self):
        # Click 4 points with Left Mouse Button
        clicks = [(100, 100), (300, 100), (300, 300), (100, 300)]
        for x, y in clicks:
            self.editor.handle_mouse(cv2.EVENT_LBUTTONDOWN, x, y, 0)

        self.assertEqual(len(self.editor.current_points), 4)

        # Finish polygon with Enter
        handled = self.editor.handle_key("\r")
        self.assertTrue(handled)
        self.assertEqual(len(self.manager.zones), 1)

        saved_zone = self.manager.get_zone(0)
        self.assertEqual(saved_zone.multiplier, 1)
        self.assertEqual(len(saved_zone.points), 4)
        # Verify points cleared and multiplier auto-advanced to 2x
        self.assertEqual(len(self.editor.current_points), 0)
        self.assertEqual(self.editor.active_multiplier, 2)

    def test_undo_and_cancel(self):
        # Add 3 points
        self.editor.add_point(10, 10)
        self.editor.add_point(20, 20)
        self.editor.add_point(30, 30)
        self.assertEqual(len(self.editor.current_points), 3)

        # Undo with 'u' key
        self.editor.handle_key("u")
        self.assertEqual(len(self.editor.current_points), 2)

        # Undo with Right Click
        self.editor.handle_mouse(cv2.EVENT_RBUTTONDOWN, 0, 0, 0)
        self.assertEqual(len(self.editor.current_points), 1)

        # Cancel with 'c' key
        self.editor.handle_key("c")
        self.assertEqual(len(self.editor.current_points), 0)

    def test_delete_last_zone_hotkey(self):
        # Add a zone
        self.editor.add_point(0, 0)
        self.editor.add_point(50, 0)
        self.editor.add_point(50, 50)
        self.editor.finish_zone()
        self.assertEqual(len(self.manager.zones), 1)

        # Delete with 'd'
        self.editor.handle_key("d")
        self.assertEqual(len(self.manager.zones), 0)

    def test_save_and_load_hotkeys(self):
        # Draw and finish zone
        self.editor.set_multiplier(3)
        self.editor.add_point(100, 100)
        self.editor.add_point(200, 100)
        self.editor.add_point(200, 200)
        self.editor.finish_zone()

        # Save with 's'
        self.editor.handle_key("s")
        self.assertTrue(os.path.exists(self.config_path))

        # Clear and load with 'l'
        self.manager.clear_zones()
        self.assertEqual(len(self.manager.zones), 0)
        self.editor.handle_key("l")
        self.assertEqual(len(self.manager.zones), 1)
        self.assertEqual(self.manager.get_zone(0).multiplier, 3)

    def test_render_ui_visuals(self):
        frame = np.full((1200, 800, 3), 220, dtype=np.uint8)
        self.editor.add_point(200, 200)
        self.editor.add_point(400, 200)
        self.editor.cursor_pos = (400, 400)

        rendered = self.editor.render_ui(frame)
        self.assertEqual(rendered.shape, frame.shape)
        self.assertEqual(rendered.dtype, np.uint8)
        self.assertFalse(np.array_equal(rendered, frame))

    def test_quit_key(self):
        # 'q' key returns False to terminate loop
        self.assertFalse(self.editor.handle_key("q"))

    def test_finish_zone_alternative_keys(self):
        # Test finishing with Space and 'n'
        for key in [" ", "n"]:
            self.editor.add_point(0, 0)
            self.editor.add_point(100, 0)
            self.editor.add_point(100, 100)
            self.editor.handle_key(key)
        self.assertEqual(len(self.manager.zones), 2)

    def test_finish_insufficient_points_returns_none(self):
        self.editor.add_point(10, 10)
        self.editor.add_point(20, 20)
        zone = self.editor.finish_zone()
        self.assertIsNone(zone)
        self.assertEqual(len(self.manager.zones), 0)
        self.assertIn("requires >= 3 points", self.editor.status_message)


class TestCalibrationPipeline(unittest.TestCase):
    """End-to-end integration tests for CalibrationPipeline."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "pipeline_config.json")
        self.sheet = generate_aruco_calibration_sheet(800, 1200, marker_size=100, margin=50)
        self.pipeline = CalibrationPipeline(
            calibrator=ArucoCalibrator(
                board_width=800,
                board_height=1200,
                marker_inset_x=100.0,
                marker_inset_y=100.0,
            ),
            config_path=self.config_path,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_end_to_end_calibration_and_scoring(self):
        # 1. Calibrate from synthetic camera frame
        success = self.pipeline.calibrate_camera_frame(self.sheet)
        self.assertTrue(success)

        # 2. Add scoring zones for 1x..6x
        for m in VALID_MULTIPLIERS:
            y1 = (m - 1) * 180 + 50
            y2 = y1 + 160
            pts = [(100.0, float(y1)), (700.0, float(y1)), (700.0, float(y2)), (100.0, float(y2))]
            self.pipeline.zone_manager.add_zone(
                ScoringZone(name=f"Zone {m}x", multiplier=m, points=pts)
            )

        # 3. Score points in board coordinates
        for m in VALID_MULTIPLIERS:
            test_y = (m - 1) * 180 + 130
            mult, name = self.pipeline.score_board_point(400, test_y)
            self.assertEqual(mult, m)
            self.assertEqual(name, f"Zone {m}x")

        # 4. Score points in camera coordinates (on sheet, marker center TL is near 100, 100)
        # Point inside Zone 1x: (400, 130)
        bx, by, mult, name = self.pipeline.score_camera_point(400, 130)
        self.assertEqual(mult, 1)
        self.assertEqual(name, "Zone 1x")

        # 5. Save and reload pipeline
        self.pipeline.save(self.config_path)
        self.assertTrue(os.path.exists(self.config_path))

        new_pipe = CalibrationPipeline(config_path=self.config_path)
        new_pipe.load(self.config_path)
        self.assertIsNotNone(new_pipe.calibrator.homography_matrix)
        self.assertEqual(len(new_pipe.zone_manager.zones), 6)

        # Verify scoring with loaded pipeline
        bx, by, mult, name = new_pipe.score_camera_point(400, 130)
        self.assertEqual(mult, 1)

    def test_process_frame_dual_views(self):
        self.pipeline.calibrate_camera_frame(self.sheet)
        self.pipeline.zone_manager.add_zone(
            ScoringZone("Zone 2x", 2, [(100, 100), (300, 100), (300, 300), (100, 300)])
        )
        warped_view, cam_view = self.pipeline.process_frame(self.sheet)
        self.assertEqual(warped_view.shape, (1200, 800, 3))
        self.assertEqual(cam_view.shape, self.sheet.shape)


if __name__ == "__main__":
    unittest.main()
