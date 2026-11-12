"""
test_app.py - Comprehensive Unit Tests for Tumblin' Dice Scoreboard HUD

Verification test command:
    python -m unittest test_app.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest

# Ensure headless Qt operation for tests
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import cv2
import numpy as np
from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QLineEdit, QPushButton

from app import (
    DEFAULT_CONFIG_PATH,
    FLICKS_PER_PLAYER_PER_ROUND,
    MAX_SUPPORTED_PLAYERS,
    MIN_SUPPORTED_PLAYERS,
    TOTAL_MAJOR_ROUNDS,
    BoardOverlayWidget,
    DeltaDieScorer,
    DeltaScoringResult,
    FlickRecord,
    GameState,
    LeaderboardWidget,
    Player,
    PlayerSetupDialog,
    TouchButton,
    TouchScoreWidget,
    TumblinDiceHUD,
    TurnBannerWidget,
    VisionPipelineThread,
)
from calibration import ScoringZone, generate_aruco_calibration_sheet
from detection import DetectedDie, create_synthetic_board
from motion import MotionConfig, MotionDetector, MotionResult, MotionState

# Shared QApplication for test session
app = QApplication.instance()
if not app:
    app = QApplication([])


class TestPlayerModel(unittest.TestCase):
    """Unit tests for the Player data model."""

    def test_player_initialization(self):
        player = Player(id=1, name="Alice", color="#EF4444")
        self.assertEqual(player.id, 1)
        self.assertEqual(player.name, "Alice")
        self.assertEqual(player.color, "#EF4444")
        self.assertEqual(len(player.flicks), TOTAL_MAJOR_ROUNDS)
        self.assertEqual(player.total_score(), 0)

    def test_round_and_total_scores(self):
        player = Player(id=2, name="Bob", color="#3B82F6")
        player.flicks[1] = [12, 6, 0, 18]  # Sum = 36
        player.flicks[2] = [4, 8, 12, 16]  # Sum = 40
        player.flicks[3] = [0, 0, 0, 0]    # Sum = 0
        player.flicks[4] = [24, 24, 18, 12]  # Sum = 78

        self.assertEqual(player.round_score(1), 36)
        self.assertEqual(player.round_score(2), 40)
        self.assertEqual(player.round_score(3), 0)
        self.assertEqual(player.round_score(4), 78)
        self.assertEqual(player.total_score(), 154)

    def test_serialization(self):
        player = Player(id=3, name="Charlie", color="#10B981")
        player.flicks[1] = [5, 10]
        data = player.to_dict()
        reconstructed = Player.from_dict(data)

        self.assertEqual(reconstructed.id, 3)
        self.assertEqual(reconstructed.name, "Charlie")
        self.assertEqual(reconstructed.color, "#10B981")
        self.assertEqual(reconstructed.round_score(1), 15)


class TestGameStateSnakeOrder(unittest.TestCase):
    """
    Unit tests specifically verifying the 4 Major Rounds snake turn order:
    - Round 1: Forward (1 -> N)
    - Round 2: Reverse (N -> 1)
    - Round 3: Forward (1 -> N)
    - Round 4: Reverse (N -> 1)
    """

    def setUp(self):
        self.players = [
            Player(id=1, name="P1", color="#EF4444"),
            Player(id=2, name="P2", color="#3B82F6"),
            Player(id=3, name="P3", color="#10B981"),
            Player(id=4, name="P4", color="#F59E0B"),
        ]
        self.state = GameState(players=self.players)

    def test_round_directions(self):
        self.assertFalse(self.state.is_round_reversed(1))
        self.assertTrue(self.state.is_round_reversed(2))
        self.assertFalse(self.state.is_round_reversed(3))
        self.assertTrue(self.state.is_round_reversed(4))

        self.assertEqual(self.state.get_round_direction_name(1), "Forward (1 ➔ N)")
        self.assertEqual(self.state.get_round_direction_name(2), "Reverse (N ➔ 1)")
        self.assertEqual(self.state.get_round_direction_name(3), "Forward (1 ➔ N)")
        self.assertEqual(self.state.get_round_direction_name(4), "Reverse (N ➔ 1)")

    def test_round_ordered_players(self):
        r1_order = [p.name for p in self.state.get_round_ordered_players(1)]
        r2_order = [p.name for p in self.state.get_round_ordered_players(2)]
        r3_order = [p.name for p in self.state.get_round_ordered_players(3)]
        r4_order = [p.name for p in self.state.get_round_ordered_players(4)]

        self.assertEqual(r1_order, ["P1", "P2", "P3", "P4"])
        self.assertEqual(r2_order, ["P4", "P3", "P2", "P1"])
        self.assertEqual(r3_order, ["P1", "P2", "P3", "P4"])
        self.assertEqual(r4_order, ["P4", "P3", "P2", "P1"])

    def test_full_round_turn_sequence_cycles(self):
        # Round 1: 4 players * 4 flicks = 16 turns
        r1_seq = self.state.get_round_turn_sequence(1)
        self.assertEqual(len(r1_seq), 16)

        # Cycle 1 (Flick 1)
        self.assertEqual([(p.name, f) for p, f in r1_seq[0:4]], [("P1", 1), ("P2", 1), ("P3", 1), ("P4", 1)])
        # Cycle 2 (Flick 2)
        self.assertEqual([(p.name, f) for p, f in r1_seq[4:8]], [("P1", 2), ("P2", 2), ("P3", 2), ("P4", 2)])
        # Cycle 3 (Flick 3)
        self.assertEqual([(p.name, f) for p, f in r1_seq[8:12]], [("P1", 3), ("P2", 3), ("P3", 3), ("P4", 3)])
        # Cycle 4 (Flick 4)
        self.assertEqual([(p.name, f) for p, f in r1_seq[12:16]], [("P1", 4), ("P2", 4), ("P3", 4), ("P4", 4)])

        # Round 2: Reversing (P4 -> P1)
        r2_seq = self.state.get_round_turn_sequence(2)
        self.assertEqual(len(r2_seq), 16)
        # Cycle 1 (Flick 1)
        self.assertEqual([(p.name, f) for p, f in r2_seq[0:4]], [("P4", 1), ("P3", 1), ("P2", 1), ("P1", 1)])
        # Cycle 4 (Flick 4)
        self.assertEqual([(p.name, f) for p, f in r2_seq[12:16]], [("P4", 4), ("P3", 4), ("P2", 4), ("P1", 4)])

    def test_two_players_snake_order(self):
        state2 = GameState(players=[Player(1, "A", "#F00"), Player(2, "B", "#00F")])
        r1 = [(p.name, f) for p, f in state2.get_round_turn_sequence(1)]
        r2 = [(p.name, f) for p, f in state2.get_round_turn_sequence(2)]

        self.assertEqual(r1[0:2], [("A", 1), ("B", 1)])
        self.assertEqual(r2[0:2], [("B", 1), ("A", 1)])

    def test_ten_players_snake_order(self):
        state10 = GameState()
        state10._init_default_players(10)
        self.assertEqual(state10.player_count, 10)

        r1 = state10.get_round_ordered_players(1)
        self.assertEqual([p.id for p in r1], list(range(1, 11)))

        r2 = state10.get_round_ordered_players(2)
        self.assertEqual([p.id for p in r2], list(range(10, 0, -1)))


class TestGameStateScoringAndProgression(unittest.TestCase):
    """Unit tests for turn recording, scoring, undo, and round transitions."""

    def setUp(self):
        self.state = GameState()
        self.state._init_default_players(3)  # 3 players: P1, P2, P3

    def test_record_flick_calculated_score(self):
        turn = self.state.get_current_turn()
        self.assertIsNotNone(turn)
        self.assertEqual(turn[0].name, "Player 1")
        self.assertEqual(turn[1], 1)

        # Pip=5, Multiplier=3 -> Score=15
        rec = self.state.record_flick(score=0, pips=5, multiplier=3)
        self.assertEqual(rec.score, 15)
        self.assertEqual(rec.player_id, 1)
        self.assertEqual(self.state.players[0].round_score(1), 15)

        # Next turn should be Player 2
        next_turn = self.state.get_current_turn()
        self.assertEqual(next_turn[0].name, "Player 2")

    def test_spacebar_miss_override(self):
        # Player 1 misses completely
        rec = self.state.record_miss()
        self.assertTrue(rec.is_miss)
        self.assertEqual(rec.score, 0)
        self.assertEqual(rec.pips, 0)
        self.assertEqual(rec.multiplier, 0)
        self.assertEqual(self.state.players[0].round_score(1), 0)

        # Next turn is Player 2
        turn2 = self.state.get_current_turn()
        self.assertEqual(turn2[0].name, "Player 2")

    def test_round_completion_and_transition(self):
        total_turns = 3 * 4  # 12 turns in round 1
        for i in range(total_turns):
            self.assertFalse(self.state.is_current_round_complete())
            self.state.record_flick(score=5)

        self.assertTrue(self.state.is_current_round_complete())
        self.assertFalse(self.state.is_game_over)

        # Transition to round 2
        advanced = self.state.advance_to_next_round()
        self.assertTrue(advanced)
        self.assertEqual(self.state.current_round, 2)
        self.assertEqual(self.state.current_turn_index, 0)

        # Round 2 active player should be Player 3 (reverse order)
        r2_turn1 = self.state.get_current_turn()
        self.assertEqual(r2_turn1[0].name, "Player 3")

    def test_full_game_playthrough_to_game_over(self):
        for r in range(1, 5):
            for t in range(self.state.total_turns_per_round):
                self.state.record_flick(score=10)
            if r < 4:
                self.assertTrue(self.state.advance_to_next_round())

        self.assertTrue(self.state.is_game_over)
        self.assertIsNone(self.state.get_current_turn())
        self.assertEqual(len(self.state.history), 3 * 4 * 4)  # 48 records

        # Attempting to record after game over raises RuntimeError
        with self.assertRaises(RuntimeError):
            self.state.record_flick(score=5)

    def test_undo_single_and_multiple(self):
        self.state.record_flick(score=10)  # P1
        self.state.record_flick(score=20)  # P2
        self.assertEqual(self.state.players[0].round_score(1), 10)
        self.assertEqual(self.state.players[1].round_score(1), 20)

        # Undo P2
        self.assertTrue(self.state.undo())
        self.assertEqual(self.state.players[1].round_score(1), 0)
        self.assertEqual(self.state.get_current_turn()[0].name, "Player 2")

        # Undo P1
        self.assertTrue(self.state.undo())
        self.assertEqual(self.state.players[0].round_score(1), 0)
        self.assertEqual(self.state.get_current_turn()[0].name, "Player 1")

        # Undo on empty history
        self.assertFalse(self.state.undo())

    def test_undo_across_round_boundary(self):
        # Complete round 1
        for _ in range(self.state.total_turns_per_round):
            self.state.record_flick(score=1)
        self.state.advance_to_next_round()
        self.assertEqual(self.state.current_round, 2)

        # Record 1 flick in round 2
        self.state.record_flick(score=6)
        self.assertEqual(self.state.players[2].round_score(2), 6)  # P3 went first

        # Undo that flick
        self.state.undo()
        self.assertEqual(self.state.current_round, 2)
        self.assertEqual(self.state.current_turn_index, 0)
        self.assertEqual(self.state.players[2].round_score(2), 0)

        # Undo the last flick of round 1
        self.state.undo()
        self.assertEqual(self.state.current_round, 1)
        self.assertEqual(self.state.current_turn_index, self.state.total_turns_per_round - 1)

    def test_manual_score_override(self):
        self.state.record_flick(score=10)  # P1
        p1 = self.state.players[0]
        self.assertEqual(p1.round_score(1), 10)

        # Correct flick 0 to 18
        success = self.state.set_flick_score(player_id=p1.id, round_num=1, flick_idx_0_based=0, new_score=18)
        self.assertTrue(success)
        self.assertEqual(p1.round_score(1), 18)

        # Invalid player or index
        self.assertFalse(self.state.set_flick_score(999, 1, 0, 5))
        self.assertFalse(self.state.set_flick_score(p1.id, 1, 99, 5))

    def test_leaderboard_and_winners(self):
        # P1 scores 10, P2 scores 25, P3 scores 15
        self.state.record_flick(score=10)
        self.state.record_flick(score=25)
        self.state.record_flick(score=15)

        board = self.state.get_leaderboard()
        self.assertEqual(board[0].name, "Player 2")
        self.assertEqual(board[1].name, "Player 3")
        self.assertEqual(board[2].name, "Player 1")

        winners = self.state.get_winners()
        self.assertEqual(len(winners), 1)
        self.assertEqual(winners[0].name, "Player 2")

    def test_reset_game(self):
        self.state.record_flick(score=10)
        self.state.record_flick(score=20)
        self.state.reset_game()

        self.assertEqual(self.state.current_round, 1)
        self.assertEqual(self.state.current_turn_index, 0)
        self.assertEqual(len(self.state.history), 0)
        self.assertEqual(self.state.players[0].total_score(), 0)

    def test_state_persistence_json(self):
        self.state.record_flick(score=12, pips=4, multiplier=3)
        self.state.record_miss()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self.state.save_to_json(tmp_path)
            loaded_state = GameState.load_from_json(tmp_path)

            self.assertEqual(loaded_state.current_round, self.state.current_round)
            self.assertEqual(loaded_state.current_turn_index, self.state.current_turn_index)
            self.assertEqual(len(loaded_state.history), 2)
            self.assertEqual(loaded_state.players[0].round_score(1), 12)
            self.assertEqual(loaded_state.players[1].round_score(1), 0)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestPyQt6TouchWidgets(unittest.TestCase):
    """
    Unit tests for PyQt6 touch controls, sizing, and widgets.
    Confirms touch target ergonomics, layout structure, and button signals.
    """

    def test_touch_button_dimensions(self):
        btn = TouchButton("Test Touch", min_height=56, min_width=64)
        self.assertGreaterEqual(btn.minimumHeight(), 48)
        self.assertGreaterEqual(btn.minimumWidth(), 48)

    def test_touch_score_widget_layout_and_sizes(self):
        score_widget = TouchScoreWidget()

        # Miss button touch target
        self.assertGreaterEqual(score_widget.btn_miss.minimumHeight(), 40)
        # Confirm button touch target
        self.assertGreaterEqual(score_widget.btn_confirm.minimumHeight(), 40)

        # Pip buttons (1..6)
        self.assertEqual(len(score_widget.pip_buttons), 6)
        for btn in score_widget.pip_buttons:
            self.assertGreaterEqual(btn.minimumHeight(), 32)
            self.assertGreaterEqual(btn.minimumWidth(), 32)

        # Multiplier buttons (1x..6x)
        self.assertEqual(len(score_widget.mult_buttons), 6)
        for btn in score_widget.mult_buttons:
            self.assertGreaterEqual(btn.minimumHeight(), 32)
            self.assertGreaterEqual(btn.minimumWidth(), 32)

    def test_touch_score_selection_and_signals(self):
        score_widget = TouchScoreWidget()
        recorded = []
        score_widget.score_confirmed.connect(lambda s, p, m: recorded.append((s, p, m)))

        # Select Pip 4, Multiplier 3x
        score_widget.select_pip(4)
        score_widget.select_multiplier(3)
        self.assertEqual(score_widget.selected_pip, 4)
        self.assertEqual(score_widget.selected_multiplier, 3)
        self.assertIn("12 PTS", score_widget.btn_confirm.text())

        # Click confirm
        QTest.mouseClick(score_widget.btn_confirm, Qt.MouseButton.LeftButton)
        self.assertEqual(len(recorded), 1)
        self.assertEqual(recorded[0], (12, 4, 3))

    def test_turn_banner_view_updates(self):
        banner = TurnBannerWidget()
        state = GameState()
        banner.update_view(state)

        self.assertIn("ROUND 1 OF 4", banner.lbl_round.text())
        self.assertIn("Forward", banner.lbl_direction.text())
        self.assertEqual(banner.lbl_player_name.text(), "Player 1")

        # Advance to round 2
        for _ in range(state.total_turns_per_round):
            state.record_flick(score=2)
        state.advance_to_next_round()
        banner.update_view(state)

        self.assertIn("ROUND 2 OF 4", banner.lbl_round.text())
        self.assertIn("Reverse", banner.lbl_direction.text())
        self.assertEqual(banner.lbl_player_name.text(), "Player 4")

    def test_leaderboard_widget_table_population(self):
        leaderboard = LeaderboardWidget()
        state = GameState()
        state.record_flick(score=18)  # P1
        state.record_flick(score=24)  # P2
        leaderboard.update_view(state)

        # Rows = 4 players
        self.assertEqual(leaderboard.table.rowCount(), 4)
        # First row should be Player 2 with 24 pts
        top_player_item = leaderboard.table.item(0, 1)
        self.assertIn("Player 2", top_player_item.text())
        top_total_item = leaderboard.table.item(0, 6)
        self.assertEqual(top_total_item.text(), "24")

    def test_board_overlay_point_in_polygon(self):
        board = BoardOverlayWidget(config_path="non_existent.json")
        # Test synthetic rectangular zone 1x: [100, 200] to [700, 400]
        self.assertTrue(board._is_point_in_polygon(400, 300, board.zones[0]["points"]))
        self.assertFalse(board._is_point_in_polygon(50, 50, board.zones[0]["points"]))

        # Add a die pin
        board.add_die_pin(400, 300, "#EF4444", 5, 2)
        self.assertEqual(len(board.active_dice), 1)
        board.clear_dice()
        self.assertEqual(len(board.active_dice), 0)

    def test_board_overlay_with_zones_config_json(self):
        # Test loading actual zones_config.json if present
        if os.path.exists(DEFAULT_CONFIG_PATH):
            board = BoardOverlayWidget(config_path=DEFAULT_CONFIG_PATH)
            self.assertEqual(board.board_width, 800)
            self.assertEqual(board.board_height, 1200)
            self.assertGreater(len(board.zones), 0)

    def test_player_setup_dialog(self):
        players = [Player(1, "Original", "#EF4444")]
        dlg = PlayerSetupDialog(players)
        dlg.name_inputs[0].setText("NewName")
        dlg._on_save()
        configured = dlg.get_configured_players()
        self.assertEqual(configured[0].name, "NewName")
        self.assertEqual(configured[0].color, "#EF4444")



class TestTumblinDiceHUDIntegration(unittest.TestCase):
    """
    End-to-End integration tests for the TumblinDiceHUD window.
    Tests user interactions: Spacebar miss override, touch controls, keyboard shortcuts.
    """

    def setUp(self):
        self.state = GameState()
        self.hud = TumblinDiceHUD(state=self.state)
        self.hud.show()
        app.processEvents()

    def tearDown(self):
        self.hud.close()
        app.processEvents()

    def test_spacebar_global_override_records_miss(self):
        # Initial turn is Player 1
        self.assertEqual(self.state.get_active_player().name, "Player 1")
        self.assertEqual(self.state.players[0].round_score(1), 0)

        # Press Spacebar key on window
        QTest.keyClick(self.hud, Qt.Key.Key_Space)
        app.processEvents()

        # Turn advanced to Player 2, and Player 1 recorded 0 pts
        self.assertEqual(self.state.get_active_player().name, "Player 2")
        self.assertEqual(self.state.players[0].round_score(1), 0)
        self.assertEqual(len(self.state.history), 1)
        self.assertTrue(self.state.history[0].is_miss)
        self.assertIn("MISS (0 pts) recorded", self.hud.lbl_status.text())

    def test_spacebar_intercepted_when_button_has_focus(self):
        # Give focus to confirm button
        self.hud.touch_controls.btn_confirm.setFocus()
        self.assertTrue(self.hud.touch_controls.btn_confirm.hasFocus())

        # Press Spacebar
        QTest.keyClick(self.hud.touch_controls.btn_confirm, Qt.Key.Key_Space)
        app.processEvents()

        # It MUST trigger miss override, not the confirm button!
        self.assertEqual(len(self.state.history), 1)
        self.assertTrue(self.state.history[0].is_miss)
        self.assertEqual(self.state.history[0].score, 0)

    def test_spacebar_not_intercepted_in_line_edit(self):
        # When typing into a text field, space should be allowed
        edit = QLineEdit(self.hud)
        edit.show()
        edit.setFocus()
        app.processEvents()

        # Type space into edit
        QTest.keyClicks(edit, "A B")
        self.assertEqual(edit.text(), "A B")
        # Game state should NOT have registered a miss
        self.assertEqual(len(self.state.history), 0)

    def test_miss_touch_button_click(self):
        # Click on-screen 0 / Miss button
        QTest.mouseClick(self.hud.touch_controls.btn_miss, Qt.MouseButton.LeftButton)
        app.processEvents()

        self.assertEqual(len(self.state.history), 1)
        self.assertTrue(self.state.history[0].is_miss)
        self.assertEqual(self.state.get_active_player().name, "Player 2")

    def test_touch_confirm_score_flow(self):
        # Select Pip 6, Multiplier 4x -> 24 pts
        self.hud.touch_controls.select_pip(6)
        self.hud.touch_controls.select_multiplier(4)
        QTest.mouseClick(self.hud.touch_controls.btn_confirm, Qt.MouseButton.LeftButton)
        app.processEvents()

        self.assertEqual(self.state.players[0].round_score(1), 24)
        self.assertEqual(self.state.get_active_player().name, "Player 2")
        self.assertEqual(len(self.hud.board_overlay.active_dice), 1)

    def test_number_key_shortcuts(self):
        # Press key '3' for 3 pips
        QTest.keyClick(self.hud, Qt.Key.Key_3)
        self.assertEqual(self.hud.touch_controls.selected_pip, 3)

        # Press Enter to confirm score
        QTest.keyClick(self.hud, Qt.Key.Key_Return)
        app.processEvents()

        # P1 scored 3 pips * 1x = 3 pts
        self.assertEqual(self.state.players[0].round_score(1), 3)

    def test_undo_via_button_and_ctrl_z(self):
        # Record a score
        self.hud.touch_controls.select_pip(5)
        self.hud.touch_controls.select_multiplier(2)
        QTest.mouseClick(self.hud.touch_controls.btn_confirm, Qt.MouseButton.LeftButton)
        app.processEvents()
        self.assertEqual(self.state.players[0].round_score(1), 10)

        # Click Undo button
        QTest.mouseClick(self.hud.touch_controls.btn_undo, Qt.MouseButton.LeftButton)
        app.processEvents()
        self.assertEqual(self.state.players[0].round_score(1), 0)
        self.assertEqual(self.state.get_active_player().name, "Player 1")

        # Record another score and undo via Ctrl+Z
        self.hud.on_miss_override()
        self.assertEqual(len(self.state.history), 1)
        QTest.keyClick(self.hud, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
        app.processEvents()
        self.assertEqual(len(self.state.history), 0)

    def test_full_game_hud_transitions(self):
        # Simulate entire 4-round game rapidly
        total_flicks = self.state.player_count * self.state.flicks_per_player * 4
        for r in range(1, 5):
            for f in range(self.state.total_turns_per_round):
                self.hud.touch_controls.select_pip(3)
                self.hud.touch_controls.select_multiplier(2)
                self.hud.touch_controls._on_confirm_clicked()
                app.processEvents()
            if r < 4:
                self.assertTrue(self.hud.touch_controls.btn_next_round.isVisible())
                QTest.mouseClick(self.hud.touch_controls.btn_next_round, Qt.MouseButton.LeftButton)
                app.processEvents()

        self.assertTrue(self.state.is_game_over)
        self.assertIn("GAME COMPLETE", self.hud.banner.lbl_round.text())

    def test_rapid_consecutive_misses(self):
        # 4 consecutive misses for the 4 players in Round 1 Cycle 1
        for expected_player in ["Player 1", "Player 2", "Player 3", "Player 4"]:
            self.assertEqual(self.state.get_active_player().name, expected_player)
            QTest.keyClick(self.hud, Qt.Key.Key_Space)
            app.processEvents()

        # Turn should now be on cycle 2, starting with Player 1 again in forward order
        self.assertEqual(self.state.get_active_player().name, "Player 1")
        self.assertEqual(self.state.get_current_flick_number(), 2)
        self.assertEqual(len(self.state.history), 4)

    def test_board_zone_click_selects_multiplier(self):
        # Simulate zone selection from board overlay signal
        self.hud._on_board_zone_clicked(5)
        self.assertEqual(self.hud.touch_controls.selected_multiplier, 5)
        self.assertIn("Selected 5x zone", self.hud.lbl_status.text())

    def test_mock_feed_initialization_and_controls(self):
        mock_hud = TumblinDiceHUD(state=GameState(), mock_feed=True)
        self.assertTrue(mock_hud.mock_feed)
        self.assertTrue(hasattr(mock_hud, "btn_mock_roll"))
        self.assertTrue(hasattr(mock_hud, "btn_mock_toggle"))

        # Test toggling mock feed
        mock_hud.toggle_mock_feed_auto()
        self.assertTrue(mock_hud.mock_auto_running)
        self.assertTrue(mock_hud.mock_timer.isActive())

        mock_hud.toggle_mock_feed_auto()
        self.assertFalse(mock_hud.mock_auto_running)
        self.assertFalse(mock_hud.mock_timer.isActive())
        mock_hud.close()

    def test_trigger_mock_flick_advances_game(self):
        mock_hud = TumblinDiceHUD(state=GameState(), mock_feed=True)
        initial_flicks = len(mock_hud.state.history)

        mock_hud.trigger_mock_flick()
        self.assertEqual(len(mock_hud.state.history), initial_flicks + 1)
        self.assertIn("[MOCK FEED]", mock_hud.lbl_status.text())
        mock_hud.close()


class TestDeltaDieScorer(unittest.TestCase):
    """Unit tests for the DeltaDieScorer baseline tracking and scoring engine."""

    def setUp(self):
        self.scorer = DeltaDieScorer(match_distance_threshold=75.0)

    def _create_mock_die(
        self,
        die_id: int,
        cx: float,
        cy: float,
        pips: int = 4,
        multiplier: int = 2,
        player_id: Optional[int] = 1,
        is_oob: bool = False,
    ) -> DetectedDie:
        return DetectedDie(
            die_id=die_id,
            center=(cx, cy),
            bbox=(int(cx - 30), int(cy - 30), 60, 60),
            pip_count=pips,
            pip_confidence=0.9,
            player_id=player_id,
            player_name=f"Player {player_id}" if player_id else None,
            zone_multiplier=multiplier,
            score=pips * multiplier,
            is_out_of_bounds=is_oob,
        )

    def test_first_flick_empty_board(self):
        die1 = self._create_mock_die(1, 400.0, 500.0, pips=5, multiplier=2, player_id=1)
        res = self.scorer.compute_delta([die1], active_player_id=1)
        self.assertFalse(res.is_miss)
        self.assertEqual(res.new_die.die_id, 1)
        self.assertEqual(res.pips, 5)
        self.assertEqual(res.multiplier, 2)
        self.assertEqual(res.score, 10)
        self.assertEqual(res.total_delta_score, 10)
        self.assertEqual(len(self.scorer.previous_dice), 1)

    def test_second_flick_identifies_new_die(self):
        die1 = self._create_mock_die(1, 400.0, 500.0, pips=5, multiplier=2, player_id=1)
        self.scorer.compute_delta([die1], active_player_id=1)

        die1_moved = self._create_mock_die(1, 402.0, 501.0, pips=5, multiplier=2, player_id=1)
        die2 = self._create_mock_die(2, 300.0, 350.0, pips=3, multiplier=3, player_id=2)

        res = self.scorer.compute_delta([die1_moved, die2], active_player_id=2)
        self.assertFalse(res.is_miss)
        self.assertEqual(res.new_die.die_id, 2)
        self.assertEqual(res.pips, 3)
        self.assertEqual(res.multiplier, 3)
        self.assertEqual(res.score, 9)
        self.assertEqual(len(self.scorer.previous_dice), 2)

    def test_gutter_miss_detection(self):
        die1 = self._create_mock_die(1, 400.0, 500.0, pips=4, multiplier=2, player_id=1)
        self.scorer.compute_delta([die1], active_player_id=1)

        die1_same = self._create_mock_die(1, 400.0, 500.0, pips=4, multiplier=2, player_id=1)
        res = self.scorer.compute_delta([die1_same], active_player_id=2)
        self.assertTrue(res.is_miss)
        self.assertIsNone(res.new_die)
        self.assertEqual(res.score, 0)
        self.assertEqual(res.pips, 0)
        self.assertEqual(res.multiplier, 0)

    def test_bumped_die_displacement_delta(self):
        die1 = self._create_mock_die(1, 400.0, 500.0, pips=4, multiplier=2, player_id=1)
        self.scorer.compute_delta([die1], active_player_id=1)

        # Die 1 bumped from 2x (8 pts) to 4x (16 pts) -> delta = +8
        die1_bumped = self._create_mock_die(1, 400.0, 750.0, pips=4, multiplier=4, player_id=1)
        die2 = self._create_mock_die(2, 350.0, 480.0, pips=6, multiplier=2, player_id=2)

        res = self.scorer.compute_delta([die1_bumped, die2], active_player_id=2)
        self.assertFalse(res.is_miss)
        self.assertEqual(res.new_die.die_id, 2)
        self.assertEqual(res.score, 12)
        self.assertEqual(len(res.bumped_dice), 1)
        self.assertEqual(res.displaced_score_delta, 8)
        self.assertEqual(res.total_delta_score, 20)

    def test_reset_round_clears_baseline(self):
        die1 = self._create_mock_die(1, 400.0, 500.0, pips=5, multiplier=2)
        self.scorer.compute_delta([die1])
        self.assertEqual(len(self.scorer.previous_dice), 1)

        self.scorer.reset_round()
        self.assertEqual(len(self.scorer.previous_dice), 0)

    def test_undo_last_flick(self):
        die1 = self._create_mock_die(1, 400.0, 500.0)
        die2 = self._create_mock_die(2, 500.0, 600.0)
        self.scorer.set_baseline([die1, die2])
        self.assertEqual(len(self.scorer.previous_dice), 2)

        self.scorer.undo_last_flick()
        self.assertEqual(len(self.scorer.previous_dice), 1)
        self.assertEqual(self.scorer.previous_dice[0].die_id, 1)

    def test_delta_result_serialization(self):
        die1 = self._create_mock_die(1, 400.0, 500.0, pips=6, multiplier=3)
        res = self.scorer.compute_delta([die1], active_player_id=1)
        data = res.to_dict()
        self.assertEqual(data["pips"], 6)
        self.assertEqual(data["multiplier"], 3)
        self.assertEqual(data["score"], 18)
        self.assertFalse(data["is_miss"])
        self.assertIn("new_die", data)


class TestVisionPipelineThread(unittest.TestCase):
    """Unit tests for the background VisionPipelineThread integrating ArUco warp, motion, and settle."""

    def test_thread_initialization_defaults(self):
        thread = VisionPipelineThread(mock_feed=True, quiescence_duration=0.8)
        self.assertTrue(thread.mock_feed)
        self.assertIsNone(thread.camera_idx)
        self.assertEqual(thread.quiescence_duration, 0.8)
        self.assertIsNotNone(thread.calibrator)
        self.assertIsNotNone(thread.motion_detector)
        self.assertIsNotNone(thread.delta_scorer)

    def test_thread_camera_mode_setup(self):
        thread = VisionPipelineThread(camera_idx=0, mock_feed=False)
        self.assertEqual(thread.camera_idx, 0)
        self.assertFalse(thread.mock_feed)

    def test_aruco_calibration_and_perspective_warp_mock_frame(self):
        thread = VisionPipelineThread(mock_feed=True)
        board_img, _ = create_synthetic_board(width=800, height=1200, zones=thread.zones)
        sheet = generate_aruco_calibration_sheet(800, 1200)
        board_img[0:180, 0:180] = sheet[0:180, 0:180]
        board_img[0:180, 620:800] = sheet[0:180, 620:800]
        board_img[1020:1200, 620:800] = sheet[1020:1200, 620:800]
        board_img[1020:1200, 0:180] = sheet[1020:1200, 0:180]

        calib_success = thread.calibrator.calibrate_from_frame(board_img)
        self.assertTrue(calib_success)
        self.assertIsNotNone(thread.calibrator.homography_matrix)

        warped = thread.calibrator.warp_frame(board_img)
        self.assertEqual(warped.shape, (1200, 800, 3))

    def test_motion_and_quiescence_settle_sequence(self):
        thread = VisionPipelineThread(mock_feed=True, quiescence_duration=0.3)
        settle_events = []
        thread.settle_event.connect(lambda d: settle_events.append(d))

        sheet = generate_aruco_calibration_sheet(800, 1200)

        # 1. Moving frames
        for i in range(3):
            t = 0.1 * i
            dice = [{"cx": 400, "cy": 300 + i * 80, "size": 60, "pips": 4, "bg_color": (30, 30, 220), "player_id": 1}]
            b, _ = create_synthetic_board(800, 1200, dice_specs=dice, zones=thread.zones)
            b[0:180, 0:180] = sheet[0:180, 0:180]
            b[0:180, 620:800] = sheet[0:180, 620:800]
            b[1020:1200, 620:800] = sheet[1020:1200, 620:800]
            b[1020:1200, 0:180] = sheet[1020:1200, 0:180]
            thread._process_single_frame(b, timestamp=t)

        self.assertEqual(len(settle_events), 0)

        # 2. Static frames (settling begins and accumulates stillness)
        final_dice = [{"cx": 400, "cy": 540, "size": 60, "pips": 4, "bg_color": (30, 30, 220), "player_id": 1}]
        b_static, _ = create_synthetic_board(800, 1200, dice_specs=final_dice, zones=thread.zones)
        b_static[0:180, 0:180] = sheet[0:180, 0:180]
        b_static[0:180, 620:800] = sheet[0:180, 620:800]
        b_static[1020:1200, 620:800] = sheet[1020:1200, 620:800]
        b_static[1020:1200, 0:180] = sheet[1020:1200, 0:180]

        # First static frame transitions MOVING -> SETTLING (motion diff relative to moving frame is still non-zero)
        thread._process_single_frame(b_static, timestamp=0.4)
        # Second static frame: identical frames so movement score drops to 0, stillness timer starts at 0.5
        thread._process_single_frame(b_static, timestamp=0.5)
        self.assertEqual(len(settle_events), 0)

        # 3. Third static frame after quiescence duration (0.5 + 0.35s = 0.85s >= 0.3s)
        thread._process_single_frame(b_static, timestamp=0.85)

        self.assertEqual(len(settle_events), 1)
        res = settle_events[0]
        self.assertFalse(res.is_miss)
        self.assertEqual(res.pips, 4)
        self.assertGreater(res.score, 0)

    def test_lifecycle_pause_resume_stop(self):
        thread = VisionPipelineThread(mock_feed=True)
        thread.start()
        self.assertTrue(thread.isRunning())

        thread.pause()
        self.assertTrue(thread._paused)
        thread.resume()
        self.assertFalse(thread._paused)

        thread.stop(1000)
        self.assertFalse(thread.isRunning())

    def test_force_settle_emits_settle_event(self):
        thread = VisionPipelineThread(mock_feed=True)
        emitted = []
        thread.settle_event.connect(lambda d: emitted.append(d))

        thread.force_settle()
        self.assertEqual(len(emitted), 1)
        self.assertIsInstance(emitted[0], DeltaScoringResult)

    def test_reset_round_and_player_context(self):
        thread = VisionPipelineThread(mock_feed=True)
        thread.set_active_player(2, "#3B82F6")
        self.assertEqual(thread._active_player_id, 2)
        self.assertEqual(thread._active_player_color, "#3B82F6")

        thread.add_mock_die(5, 2, 300, 400, 2)
        self.assertEqual(len(thread._mock_placed_dice), 1)

        thread.reset_round()
        self.assertEqual(len(thread._mock_placed_dice), 0)
        self.assertEqual(len(thread.delta_scorer.previous_dice), 0)


class TestBoardOverlayWidgetCamera(unittest.TestCase):
    """Unit tests verifying live camera frame rendering in BoardOverlayWidget."""

    def test_set_camera_frame_updates_qimage(self):
        overlay = BoardOverlayWidget()
        frame = np.full((1200, 800, 3), 120, dtype=np.uint8)

        self.assertIsNone(overlay.latest_frame)
        self.assertIsNone(overlay.latest_qimage)

        overlay.set_camera_frame(frame)
        self.assertIsNotNone(overlay.latest_frame)
        self.assertIsNotNone(overlay.latest_qimage)
        self.assertFalse(overlay.latest_qimage.isNull())
        self.assertEqual(overlay.latest_qimage.width(), 800)
        self.assertEqual(overlay.latest_qimage.height(), 1200)

    def test_paint_event_with_and_without_frame(self):
        overlay = BoardOverlayWidget()
        overlay.resize(400, 600)
        overlay.show()
        app.processEvents()

        overlay.update()
        app.processEvents()

        frame = np.zeros((1200, 800, 3), dtype=np.uint8)
        overlay.set_camera_frame(frame)
        overlay.update()
        app.processEvents()
        overlay.close()


class TestTumblinDiceHUDVisionIntegration(unittest.TestCase):
    """Integration tests verifying HUD reaction to vision thread events."""

    def test_hud_camera_mode_initialization(self):
        state = GameState()
        hud = TumblinDiceHUD(state=state, camera_idx=9999, quiescence_duration=1.0)
        self.assertEqual(hud.camera_idx, 9999)
        self.assertIsNotNone(hud.vision_thread)
        self.assertTrue(hasattr(hud, "lbl_cam_badge"))
        self.assertTrue(hasattr(hud, "btn_force_settle"))
        hud.close()

    def test_hud_mock_feed_initialization(self):
        state = GameState()
        hud = TumblinDiceHUD(state=state, mock_feed=True)
        self.assertTrue(hud.mock_feed)
        self.assertIsNotNone(hud.vision_thread)
        self.assertTrue(hud.vision_thread.isRunning())
        hud.close()

    def test_on_vision_settle_scores_flick(self):
        state = GameState()
        hud = TumblinDiceHUD(state=state, mock_feed=False)

        mock_die = DetectedDie(
            die_id=1,
            center=(400.0, 600.0),
            bbox=(370, 570, 60, 60),
            pip_count=5,
            pip_confidence=0.95,
            player_id=1,
            player_name="Player 1",
            zone_multiplier=3,
            score=15,
        )
        delta = DeltaScoringResult(
            new_die=mock_die,
            pips=5,
            multiplier=3,
            score=15,
            is_miss=False,
            total_delta_score=15,
        )

        hud.on_vision_settle(delta)
        self.assertEqual(state.players[0].round_score(1), 15)
        self.assertEqual(state.get_active_player().name, "Player 2")
        self.assertIn("15 pts", hud.lbl_status.text())
        hud.close()

    def test_on_vision_settle_gutter_miss(self):
        state = GameState()
        hud = TumblinDiceHUD(state=state, mock_feed=False)

        delta = DeltaScoringResult(
            new_die=None,
            pips=0,
            multiplier=0,
            score=0,
            is_miss=True,
        )

        hud.on_vision_settle(delta)
        self.assertEqual(state.players[0].round_score(1), 0)
        self.assertEqual(state.get_active_player().name, "Player 2")
        self.assertTrue(state.history[0].is_miss)
        self.assertIn("Gutter Miss (0 pts)", hud.lbl_status.text())
        hud.close()

    def test_on_motion_status_updates(self):
        hud = TumblinDiceHUD(state=GameState(), mock_feed=False)

        m_moving = MotionResult(
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
        hud.on_motion_status(m_moving)
        self.assertIn("ROLLING...", hud.lbl_status.text())

        m_settling = MotionResult(
            state=MotionState.SETTLING,
            movement_score=0.0,
            motion_detected=False,
            is_settled=False,
            is_scoreboard_frozen=True,
            just_settled=False,
            stillness_duration=0.6,
            quiescence_progress=0.5,
            timestamp=1.6,
        )
        hud.on_motion_status(m_settling)
        self.assertIn("SETTLING...", hud.lbl_status.text())
        self.assertIn("50%", hud.lbl_status.text())
        hud.close()

    def test_calibration_badge_signal(self):
        hud = TumblinDiceHUD(state=GameState(), mock_feed=False)
        hud.on_calibration_status(True)
        self.assertIn("Warped", hud.lbl_cal_badge.text())
        hud.on_calibration_status(False)
        self.assertIn("Scan", hud.lbl_cal_badge.text())
        hud.close()

    def test_advance_round_resets_vision_baseline(self):
        state = GameState()
        hud = TumblinDiceHUD(state=state, mock_feed=True)
        hud.vision_thread.add_mock_die(4, 2, 350, 400, 1)
        self.assertEqual(len(hud.vision_thread._mock_placed_dice), 1)

        # Complete round 1
        for _ in range(state.total_turns_per_round):
            state.record_flick(score=5)
        hud.advance_round()

        self.assertEqual(len(hud.vision_thread._mock_placed_dice), 0)
        hud.close()


class TestSurfacePro9ThreeColumnLayout(unittest.TestCase):
    """Unit tests for the 3-column layout tailored for Microsoft Surface Pro 9."""

    def setUp(self):
        self.state = GameState()
        self.hud = TumblinDiceHUD(state=self.state)
        self.hud.show()
        app.processEvents()

    def tearDown(self):
        self.hud.close()
        app.processEvents()

    def test_three_column_structure(self):
        # 1. Main horizontal splitter exists and has 3 columns
        self.assertIsNotNone(self.hud.main_splitter)
        self.assertEqual(self.hud.main_splitter.count(), 3)
        self.assertEqual(self.hud.main_splitter.orientation(), Qt.Orientation.Horizontal)

        # 2. Left column contains Leaderboard & Match Status (Banner + Leaderboard)
        self.assertIsNotNone(self.hud.left_column)
        self.assertEqual(self.hud.banner.parent(), self.hud.left_column)
        self.assertEqual(self.hud.leaderboard.parent(), self.hud.left_column)

        # 3. Center column contains BoardOverlayWidget
        self.assertIsNotNone(self.hud.center_column)
        self.assertEqual(self.hud.board_overlay.parent(), self.hud.center_column)

        # 4. Right column contains vertical TouchScoreWidget
        self.assertIsNotNone(self.hud.right_column)
        self.assertEqual(self.hud.touch_controls.parent(), self.hud.right_column)

    def test_board_overlay_preserves_two_to_three_aspect_ratio(self):
        # Check sizeHint aspect ratio is 2:3
        hint = self.hud.board_overlay.sizeHint()
        self.assertGreater(hint.width(), 0)
        self.assertGreater(hint.height(), 0)
        # width : height = 2 : 3 -> width * 3 == height * 2
        self.assertEqual(hint.width() * 3, hint.height() * 2)

        # Check coordinate mapping preserves 2:3 ratio
        self.assertEqual(self.hud.board_overlay.board_width, 800)
        self.assertEqual(self.hud.board_overlay.board_height, 1200)

    def test_right_column_vertical_touch_keypad_ergonomics(self):
        touch = self.hud.touch_controls

        # Giant 0 / Miss button (>= 74px min height for touch)
        self.assertGreaterEqual(touch.btn_miss.minimumHeight(), 74)
        self.assertIn("0 / MISS", touch.btn_miss.text())

        # Giant Confirm button (>= 74px min height for touch)
        self.assertGreaterEqual(touch.btn_confirm.minimumHeight(), 74)
        self.assertIn("CONFIRM", touch.btn_confirm.text())

        # Pip buttons (1..6) in grid
        self.assertEqual(len(touch.pip_buttons), 6)
        for btn in touch.pip_buttons:
            self.assertGreaterEqual(btn.minimumHeight(), 48)
            self.assertGreaterEqual(btn.minimumWidth(), 48)

        # Multiplier buttons (1x..6x) in grid
        self.assertEqual(len(touch.mult_buttons), 6)
        for btn in touch.mult_buttons:
            self.assertGreaterEqual(btn.minimumHeight(), 48)
            self.assertGreaterEqual(btn.minimumWidth(), 48)

        # Live calculation preview
        self.assertIsNotNone(touch.lbl_calc_preview)
        touch.select_pip(5)
        touch.select_multiplier(4)
        self.assertIn("20 pts", touch.lbl_calc_preview.text().lower())


class TestStartLaunchZoneEnforcement(unittest.TestCase):
    """
    Unit tests confirming that the 0x Start/Launch Zone at the top of the board is
    defined, and dice resting in or placed inside it are completely ignored by the CV scoring engine:
    never triggering settle events, score calculations, or turn advances.
    """

    def test_launch_zone_definition_and_detection(self):
        thread = VisionPipelineThread(mock_feed=True)
        # Verify a launch zone is present in thread.zones
        launch_zones = [z for z in thread.zones if getattr(z, "is_launch_zone", False)]
        self.assertGreaterEqual(len(launch_zones), 1)
        lz = launch_zones[0]
        self.assertEqual(lz.multiplier, 0)
        self.assertTrue(lz.is_launch_zone)

        # Die placed in launch zone (e.g. x=400, y=100)
        die_in_launch = DetectedDie(
            die_id=1,
            center=(400.0, 100.0),
            bbox=(370, 70, 60, 60),
            pip_count=5,
            pip_confidence=0.95,
            player_id=1,
        )
        scored = thread.zone_scorer.score_die(die_in_launch)
        self.assertTrue(scored.is_launch_zone)
        self.assertEqual(scored.zone_multiplier, 0)
        self.assertEqual(scored.score, 0)
        self.assertTrue(scored.is_out_of_bounds)

        # Batch scoring excludes launch zone die from active dice and player tally
        res = thread.zone_scorer.score_dice([scored])
        self.assertEqual(res.active_dice_count, 0)
        self.assertEqual(res.total_board_score, 0)
        self.assertEqual(len(res.launch_zone_dice), 1)

    def test_delta_scorer_ignores_launch_zone_dice(self):
        scorer = DeltaDieScorer()
        die_launch = DetectedDie(
            die_id=1,
            center=(400.0, 100.0),
            bbox=(370, 70, 60, 60),
            pip_count=6,
            pip_confidence=0.98,
            player_id=1,
            zone_multiplier=0,
            is_out_of_bounds=True,
            is_launch_zone=True,
            score=0,
        )
        res = scorer.compute_delta([die_launch], active_player_id=1)
        # Ignored: no valid die detected on scoring board
        self.assertTrue(res.is_miss)
        self.assertIsNone(res.new_die)
        self.assertEqual(res.score, 0)

    def test_vision_pipeline_suppresses_settle_when_die_in_launch_zone(self):
        thread = VisionPipelineThread(mock_feed=True, quiescence_duration=0.3)
        settle_events = []
        thread.settle_event.connect(lambda d: settle_events.append(d))

        sheet = generate_aruco_calibration_sheet(800, 1200)

        # Create frame where die is resting in the 0x Start/Launch zone (cy = 240, inside launch polygon [0..300])
        dice_in_launch = [{"cx": 400, "cy": 240, "size": 60, "pips": 5, "bg_color": (30, 30, 220), "player_id": 1}]
        board, _ = create_synthetic_board(800, 1200, dice_specs=dice_in_launch, zones=thread.zones)
        board[0:180, 0:180] = sheet[0:180, 0:180]
        board[0:180, 620:800] = sheet[0:180, 620:800]
        board[1020:1200, 620:800] = sheet[1020:1200, 620:800]
        board[1020:1200, 0:180] = sheet[1020:1200, 0:180]

        # Feed frame through pipeline and force settle
        thread._process_single_frame(board, timestamp=1.0)
        thread.force_settle()

        # MUST NOT trigger settle event!
        self.assertEqual(len(settle_events), 0)

    def test_vision_pipeline_emits_settle_when_die_in_scoring_zone(self):
        thread = VisionPipelineThread(mock_feed=True, quiescence_duration=0.3)
        settle_events = []
        thread.settle_event.connect(lambda d: settle_events.append(d))

        sheet = generate_aruco_calibration_sheet(800, 1200)

        # Die flicked down into Zone 2x (cy = 500)
        dice_in_zone = [{"cx": 400, "cy": 500, "size": 60, "pips": 4, "bg_color": (30, 30, 220), "player_id": 1}]
        board, _ = create_synthetic_board(800, 1200, dice_specs=dice_in_zone, zones=thread.zones)
        board[0:180, 0:180] = sheet[0:180, 0:180]
        board[0:180, 620:800] = sheet[0:180, 620:800]
        board[1020:1200, 620:800] = sheet[1020:1200, 620:800]
        board[1020:1200, 0:180] = sheet[1020:1200, 0:180]

        thread._process_single_frame(board, timestamp=1.0)
        thread.force_settle()

        # MUST trigger settle event since die is on scoring board
        self.assertEqual(len(settle_events), 1)
        res = settle_events[0]
        self.assertFalse(res.is_miss)
        self.assertEqual(res.pips, 4)
        self.assertGreater(res.score, 0)


class TestPlayerSetupDialogAndColorDropper(unittest.TestCase):
    """Unit tests for the upgraded PlayerSetupDialog with dynamic player count (1-10) and interactive video color dropper."""

    def test_dynamic_player_count_add_and_remove(self):
        initial = [Player(1, "P1", "#EF4444"), Player(2, "P2", "#3B82F6")]
        dlg = PlayerSetupDialog(initial)

        # Initial 2 players
        self.assertEqual(len(dlg.players), 2)
        self.assertEqual(len(dlg.name_inputs), 2)

        # Add players up to 10
        for _ in range(8):
            dlg.add_player()
        self.assertEqual(len(dlg.players), 10)
        self.assertFalse(dlg.btn_add_player.isEnabled())

        # Attempt to add 11th player should be rejected
        dlg.add_player()
        self.assertEqual(len(dlg.players), 10)

        # Remove players down to 1
        for _ in range(9):
            dlg.remove_player(0)
        self.assertEqual(len(dlg.players), 1)

        # Attempt to remove last player should be rejected (minimum 1 player)
        dlg.remove_player(0)
        self.assertEqual(len(dlg.players), 1)

    def test_interactive_video_color_dropper_sampling(self):
        initial = [Player(1, "Player 1", "#EF4444"), Player(2, "Player 2", "#3B82F6")]
        dlg = PlayerSetupDialog(initial)

        # Generate test synthetic frame with a known blue patch in center
        test_frame = np.full((400, 600, 3), 40, dtype=np.uint8)
        # Draw a bright cyan die at (300, 200)
        test_frame[180:220, 280:320] = (255, 200, 30)  # BGR
        dlg.video_dropper.set_frame(test_frame)

        # Select Player 2
        dlg.select_player(1)
        self.assertEqual(dlg.selected_player_index, 1)

        # Click at (300, 200) on the video frame
        res = dlg.video_dropper.sample_at_point(300, 200)
        self.assertIsNotNone(res)
        hsv, bgr, hex_code = res

        # Assigned to Player 2
        self.assertEqual(dlg.players[1].color, hex_code)
        self.assertIsNotNone(dlg.players[1].hsv_mean)

    def test_auto_sample_die_in_reticle(self):
        initial = [Player(1, "Player 1", "#EF4444")]
        dlg = PlayerSetupDialog(initial)

        # Synthetic frame with green center
        test_frame = np.full((400, 600, 3), 30, dtype=np.uint8)
        test_frame[180:220, 280:320] = (50, 205, 50)  # Lime Green BGR
        dlg.video_dropper.set_frame(test_frame)

        # Trigger auto-sample
        dlg._on_auto_sample_clicked()
        self.assertIn("assigned to", dlg.lbl_sampled_info.text().lower())
        self.assertNotEqual(dlg.players[0].color, "#EF4444")


class TestCommandLineInterface(unittest.TestCase):
    """Unit tests for the CLI entrypoint arguments."""

    def test_cli_help_flag(self):
        import subprocess
        res = subprocess.run(
            [sys.executable, "app.py", "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("--camera", res.stdout)
        self.assertIn("--mock-feed", res.stdout)
        self.assertIn("--demo", res.stdout)
        self.assertIn("--quiescence", res.stdout)

    def test_cli_mock_feed_help(self):
        import subprocess
        res = subprocess.run(
            [sys.executable, "app.py", "--mock-feed", "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("--mock-feed", res.stdout)


if __name__ == "__main__":
    unittest.main()
