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
import unittest

# Ensure headless Qt operation for tests
os.environ["QT_QPA_PLATFORM"] = "offscreen"

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
    FlickRecord,
    GameState,
    LeaderboardWidget,
    Player,
    PlayerSetupDialog,
    TouchButton,
    TouchScoreWidget,
    TumblinDiceHUD,
    TurnBannerWidget,
)

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

        # Miss button must have a massive touch target (>= 72px)
        self.assertGreaterEqual(score_widget.btn_miss.minimumHeight(), 72)
        # Confirm button must have large touch target (>= 60px)
        self.assertGreaterEqual(score_widget.btn_confirm.minimumHeight(), 60)

        # Pip buttons (1..6) must have minimum touch dimension >= 48px
        self.assertEqual(len(score_widget.pip_buttons), 6)
        for btn in score_widget.pip_buttons:
            self.assertGreaterEqual(btn.minimumHeight(), 48)
            self.assertGreaterEqual(btn.minimumWidth(), 48)

        # Multiplier buttons (1x..6x)
        self.assertEqual(len(score_widget.mult_buttons), 6)
        for btn in score_widget.mult_buttons:
            self.assertGreaterEqual(btn.minimumHeight(), 48)
            self.assertGreaterEqual(btn.minimumWidth(), 48)

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
        top_total_item = leaderboard.table.item(0, 7)
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



if __name__ == "__main__":
    unittest.main()
