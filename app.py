"""
app.py - Surface-Optimized PyQt6 Live Scoreboard HUD for Tumblin' Dice

Features:
- Full 4 Major Rounds game state machine with 4 flicks per player per round.
- Reversing snake turn order across major rounds:
    Round 1: Forward (1 -> N)
    Round 2: Reverse (N -> 1)
    Round 3: Forward (1 -> N)
    Round 4: Reverse (N -> 1)
- Surface tablet-optimized touch interface with large touch targets (minimum 48px-76px).
- Global Spacebar shortcut and prominent on-screen touch button for instant 0 / Miss override.
- Live sidebar leaderboard ranking players by cumulative total score and round breakdowns.
- Visual Turn Hero banner with player color badges, direction indicators, and flick counters.
- Interactive Tumblin' Dice board & scoring zone visualizer loading zones_config.json.
- Full Undo/Redo history, manual score override, and persistence (save/load state).
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from PyQt6.QtCore import (
    QEvent,
    QObject,
    QPointF,
    QRectF,
    QSize,
    Qt,
    QThread,
    QTimer,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QIcon,
    QImage,
    QKeyEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

from calibration import (
    DEFAULT_BOARD_HEIGHT,
    DEFAULT_BOARD_WIDTH,
    ArucoCalibrator,
    ScoringZone,
    ZoneManager,
    generate_aruco_calibration_sheet,
)
from detection import (
    DEFAULT_PLAYER_ROSTER,
    D6PipCounter,
    DetectedDie,
    DiceDetector,
    DiceScoringResult,
    DetectionPipeline,
    HSVColorSampler,
    PlayerProfileManager,
    ZoneScorer,
    create_synthetic_board,
    create_synthetic_die,
)
from motion import (
    MotionConfig,
    MotionDetector,
    MotionResult,
    MotionState,
)

# ============================================================================
# Game Constants & Defaults
# ============================================================================

DEFAULT_CONFIG_PATH = "zones_config.json"
TOTAL_MAJOR_ROUNDS = 4
FLICKS_PER_PLAYER_PER_ROUND = 4
MAX_SUPPORTED_PLAYERS = 10
MIN_SUPPORTED_PLAYERS = 1

DEFAULT_PLAYER_PALETTE = [
    ("#EF4444", "Crimson Red"),
    ("#3B82F6", "Cobalt Blue"),
    ("#10B981", "Emerald Green"),
    ("#F59E0B", "Amber Gold"),
    ("#8B5CF6", "Royal Purple"),
    ("#EC4899", "Hot Pink"),
    ("#06B6D4", "Electric Cyan"),
    ("#F97316", "Vibrant Orange"),
    ("#84CC16", "Lime Green"),
    ("#E2E8F0", "Pure White"),
]

SURFACE_QSS = """
/* Surface Tablet Dark Broadcast Theme */
QWidget {
    background-color: #0F172A;
    color: #F8FAFC;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    font-size: 15px;
}

QFrame.cardFrame {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 12px;
}

QLabel {
    background-color: transparent;
}

/* Touch-optimized Buttons */
QPushButton {
    background-color: #334155;
    color: #F8FAFC;
    border: 2px solid #475569;
    border-radius: 10px;
    padding: 10px 16px;
    font-size: 16px;
    font-weight: 600;
    min-height: 48px;
}

QPushButton:hover {
    background-color: #475569;
    border-color: #64748B;
}

QPushButton:pressed {
    background-color: #1E293B;
    border-color: #38BDF8;
}

QPushButton:checked {
    background-color: #2563EB;
    border-color: #60A5FA;
    color: #FFFFFF;
}

/* Big Miss Button */
QPushButton#missButton {
    background-color: #DC2626;
    color: #FFFFFF;
    border: 2px solid #EF4444;
    border-radius: 14px;
    font-size: 22px;
    font-weight: 800;
    min-height: 74px;
    letter-spacing: 1px;
}

QPushButton#missButton:hover {
    background-color: #B91C1C;
    border-color: #F87171;
}

QPushButton#missButton:pressed {
    background-color: #991B1B;
    border-color: #FCA5A5;
}

/* Confirm Score Button */
QPushButton#confirmButton {
    background-color: #16A34A;
    color: #FFFFFF;
    border: 2px solid #22C55E;
    border-radius: 12px;
    font-size: 19px;
    font-weight: 700;
    min-height: 60px;
}

QPushButton#confirmButton:hover {
    background-color: #15803D;
    border-color: #4ADE80;
}

QPushButton#confirmButton:pressed {
    background-color: #166534;
    border-color: #86EFAC;
}

/* Touch Keypad Pip & Multiplier Buttons */
QPushButton.gridSelectorBtn {
    font-size: 20px;
    font-weight: bold;
    min-width: 52px;
    min-height: 52px;
    border-radius: 10px;
}

/* Leaderboard Table */
QTableWidget {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 10px;
    gridline-color: #334155;
    selection-background-color: #2563EB;
    selection-color: #FFFFFF;
    font-size: 15px;
}

QHeaderView::section {
    background-color: #0F172A;
    color: #94A3B8;
    font-weight: 700;
    font-size: 14px;
    border: none;
    border-bottom: 2px solid #334155;
    padding: 10px 4px;
}

QScrollBar:vertical {
    border: none;
    background: #0F172A;
    width: 14px;
    margin: 0px;
    border-radius: 7px;
}

QScrollBar::handle:vertical {
    background: #475569;
    min-height: 30px;
    border-radius: 7px;
}

QScrollBar::handle:vertical:hover {
    background: #64748B;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""

# ============================================================================
# Core Data Models & State Machine
# ============================================================================

@dataclass
class FlickRecord:
    """Record of a single flick action taken during the game."""
    round_num: int
    player_id: int
    flick_index: int       # 1..4 (flick cycle within this major round)
    pips: int              # 1..6 or 0
    multiplier: int        # 1..6 or 0
    score: int             # pips * multiplier
    is_miss: bool          # True if 0/miss override
    turn_index: int        # index in round turn sequence
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FlickRecord:
        return cls(**data)


@dataclass
class Player:
    """Represents a Tumblin' Dice player with assigned dice color and scores."""
    id: int
    name: str
    color: str  # Hex code e.g. "#EF4444"
    flicks: Dict[int, List[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.flicks:
            self.flicks = {r: [] for r in range(1, TOTAL_MAJOR_ROUNDS + 1)}
        else:
            # Ensure int keys
            self.flicks = {int(k): list(v) for k, v in self.flicks.items()}
            for r in range(1, TOTAL_MAJOR_ROUNDS + 1):
                if r not in self.flicks:
                    self.flicks[r] = []

    def round_score(self, round_num: int) -> int:
        """Returns the sum of all flick scores recorded for a specific major round."""
        return sum(self.flicks.get(round_num, []))

    def total_score(self) -> int:
        """Returns cumulative total score across all major rounds."""
        return sum(sum(scores) for scores in self.flicks.values())

    def flicks_for_round(self, round_num: int) -> List[int]:
        """Returns list of flick scores for the specified round."""
        return self.flicks.get(round_num, [])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "color": self.color,
            "flicks": {str(k): v for k, v in self.flicks.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Player:
        flicks_raw = data.get("flicks", {})
        flicks = {int(k): list(v) for k, v in flicks_raw.items()}
        return cls(
            id=int(data["id"]),
            name=str(data["name"]),
            color=str(data["color"]),
            flicks=flicks,
        )


class GameState:
    """
    Pure Python Game Engine for Tumblin' Dice.
    
    Rules & Features:
    - 4 Major Rounds total.
    - Each player flicks 4 times per Major Round.
    - Reversing Snake Draft Order across major rounds:
        - Major Round 1: Forward (1 -> N)
        - Major Round 2: Reverse (N -> 1)
        - Major Round 3: Forward (1 -> N)
        - Major Round 4: Reverse (N -> 1)
    - Full undo/redo history and state persistence.
    """

    def __init__(
        self,
        players: Optional[List[Player]] = None,
        total_rounds: int = TOTAL_MAJOR_ROUNDS,
        flicks_per_player: int = FLICKS_PER_PLAYER_PER_ROUND,
    ) -> None:
        self.total_rounds = total_rounds
        self.flicks_per_player = flicks_per_player
        self.players: List[Player] = []
        self.current_round: int = 1
        self.current_turn_index: int = 0
        self.history: List[FlickRecord] = []
        self.is_game_over: bool = False

        if players:
            self.players = list(players)
        else:
            self._init_default_players(4)

    def _init_default_players(self, count: int = 4) -> None:
        """Initializes default players with distinct high-contrast colors."""
        self.players.clear()
        count = max(MIN_SUPPORTED_PLAYERS, min(count, MAX_SUPPORTED_PLAYERS))
        for i in range(count):
            hex_color, _ = DEFAULT_PLAYER_PALETTE[i % len(DEFAULT_PLAYER_PALETTE)]
            self.players.append(
                Player(
                    id=i + 1,
                    name=f"Player {i + 1}",
                    color=hex_color,
                )
            )

    @property
    def player_count(self) -> int:
        return len(self.players)

    @property
    def total_turns_per_round(self) -> int:
        return self.player_count * self.flicks_per_player

    def is_round_reversed(self, round_num: int) -> bool:
        """
        Determines turn order direction for a given round.
        Snake draft order:
        Round 1: Forward (1 -> N)
        Round 2: Reverse (N -> 1)
        Round 3: Forward (1 -> N)
        Round 4: Reverse (N -> 1)
        """
        return (round_num % 2 == 0)

    def get_round_direction_name(self, round_num: int) -> str:
        """Returns human-readable snake direction string."""
        if self.is_round_reversed(round_num):
            return "Reverse (N ➔ 1)"
        return "Forward (1 ➔ N)"

    def get_round_ordered_players(self, round_num: int) -> List[Player]:
        """Returns players ordered according to the snake draft rule for round_num."""
        if self.is_round_reversed(round_num):
            return list(reversed(self.players))
        return list(self.players)

    def get_round_turn_sequence(self, round_num: int) -> List[Tuple[Player, int]]:
        """
        Generates the full sequence of turns for a major round.
        Returns list of (player, flick_number) where flick_number is 1..4.
        """
        ordered = self.get_round_ordered_players(round_num)
        sequence: List[Tuple[Player, int]] = []
        for flick_cycle in range(1, self.flicks_per_player + 1):
            for player in ordered:
                sequence.append((player, flick_cycle))
        return sequence

    def get_current_turn(self) -> Optional[Tuple[Player, int]]:
        """
        Returns (active_player, flick_number) for the current turn,
        or None if the game has ended.
        """
        if self.is_game_over or self.current_round > self.total_rounds:
            return None
        seq = self.get_round_turn_sequence(self.current_round)
        if 0 <= self.current_turn_index < len(seq):
            return seq[self.current_turn_index]
        return None

    def get_active_player(self) -> Optional[Player]:
        """Returns the currently active Player, or None if game over."""
        turn = self.get_current_turn()
        return turn[0] if turn else None

    def get_current_flick_number(self) -> int:
        """Returns flick cycle number (1..4) for current turn, or 0 if game over."""
        turn = self.get_current_turn()
        return turn[1] if turn else 0

    def is_current_round_complete(self) -> bool:
        """Returns True if all players have completed all flicks for the current round."""
        return self.current_turn_index >= self.total_turns_per_round

    def record_flick(
        self,
        score: int,
        pips: int = 0,
        multiplier: int = 0,
        is_miss: bool = False,
    ) -> FlickRecord:
        """
        Records the result of the active player's flick and advances turn.
        Calculates score = pips * multiplier if pips and multiplier are provided and score is 0.
        """
        if self.is_game_over:
            raise RuntimeError("Cannot record flick: game is already over")

        turn = self.get_current_turn()
        if not turn:
            raise RuntimeError("Cannot record flick: no active turn")

        player, flick_num = turn
        if is_miss:
            final_score = 0
            pips = 0
            multiplier = 0
        elif score == 0 and pips > 0 and multiplier > 0:
            final_score = pips * multiplier
        else:
            final_score = max(0, score)

        record = FlickRecord(
            round_num=self.current_round,
            player_id=player.id,
            flick_index=flick_num,
            pips=pips,
            multiplier=multiplier,
            score=final_score,
            is_miss=is_miss,
            turn_index=self.current_turn_index,
        )
        self.history.append(record)

        # Store score in player's round flick list
        player.flicks[self.current_round].append(final_score)

        # Advance turn index
        self.current_turn_index += 1

        # Check if round completed
        if self.current_turn_index >= self.total_turns_per_round:
            if self.current_round >= self.total_rounds:
                self.is_game_over = True
            else:
                # Round ends; waiting for explicit round progression or auto-advance
                pass

        return record

    def record_miss(self) -> FlickRecord:
        """
        Instant 0 / Miss override shortcut (triggered by Spacebar or touch button).
        Records 0 points and advances to next player.
        """
        return self.record_flick(score=0, pips=0, multiplier=0, is_miss=True)

    def advance_to_next_round(self) -> bool:
        """
        Advances from a completed round to the next major round.
        Returns True if advanced, False if game is over or round not complete.
        """
        if self.is_game_over:
            return False
        if not self.is_current_round_complete():
            return False
        if self.current_round >= self.total_rounds:
            self.is_game_over = True
            return False

        self.current_round += 1
        self.current_turn_index = 0
        return True

    def undo(self) -> bool:
        """
        Rolls back the most recent flick action.
        Returns True if undone, False if history was empty.
        """
        if not self.history:
            return False

        last_record = self.history.pop()
        player = next((p for p in self.players if p.id == last_record.player_id), None)
        if player and player.flicks.get(last_record.round_num):
            player.flicks[last_record.round_num].pop()

        self.current_round = last_record.round_num
        self.current_turn_index = last_record.turn_index
        self.is_game_over = False
        return True

    def set_flick_score(
        self,
        player_id: int,
        round_num: int,
        flick_idx_0_based: int,
        new_score: int,
    ) -> bool:
        """
        Allows manual correction / override of an existing recorded flick.
        """
        player = next((p for p in self.players if p.id == player_id), None)
        if not player:
            return False
        flick_list = player.flicks.get(round_num)
        if flick_list is None or flick_idx_0_based >= len(flick_list) or flick_idx_0_based < 0:
            return False

        flick_list[flick_idx_0_based] = max(0, new_score)
        return True

    def get_leaderboard(self) -> List[Player]:
        """
        Returns players sorted by cumulative total score descending.
        Breaks ties using current round score, then player id.
        """
        return sorted(
            self.players,
            key=lambda p: (p.total_score(), p.round_score(self.current_round), -p.id),
            reverse=True,
        )

    def get_winners(self) -> List[Player]:
        """Returns the player(s) with the highest cumulative total score."""
        if not self.players:
            return []
        board = self.get_leaderboard()
        top_score = board[0].total_score()
        return [p for p in board if p.total_score() == top_score]

    def reset_game(self, players: Optional[List[Player]] = None) -> None:
        """Resets the entire game state for a new match."""
        if players is not None:
            self.players = list(players)
        else:
            for p in self.players:
                p.flicks = {r: [] for r in range(1, self.total_rounds + 1)}
        self.current_round = 1
        self.current_turn_index = 0
        self.history.clear()
        self.is_game_over = False

    def to_dict(self) -> Dict[str, Any]:
        """Serializes game state to dictionary for JSON persistence."""
        return {
            "total_rounds": self.total_rounds,
            "flicks_per_player": self.flicks_per_player,
            "current_round": self.current_round,
            "current_turn_index": self.current_turn_index,
            "is_game_over": self.is_game_over,
            "players": [p.to_dict() for p in self.players],
            "history": [h.to_dict() for h in self.history],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GameState:
        """Reconstructs game state from dictionary."""
        players = [Player.from_dict(p) for p in data.get("players", [])]
        state = cls(
            players=players,
            total_rounds=data.get("total_rounds", TOTAL_MAJOR_ROUNDS),
            flicks_per_player=data.get("flicks_per_player", FLICKS_PER_PLAYER_PER_ROUND),
        )
        state.current_round = int(data.get("current_round", 1))
        state.current_turn_index = int(data.get("current_turn_index", 0))
        state.is_game_over = bool(data.get("is_game_over", False))
        state.history = [FlickRecord.from_dict(h) for h in data.get("history", [])]
        return state

    def save_to_json(self, filepath: str) -> None:
        """Saves current state to a JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_json(cls, filepath: str) -> GameState:
        """Loads state from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


# ============================================================================
# Computer Vision Delta Scoring & Background Video Pipeline
# ============================================================================

@dataclass
class DeltaScoringResult:
    """
    Result container produced when comparing the settled board state against
    prior resting dice to isolate the active player's newly flicked die and any bumps.
    """
    new_die: Optional[DetectedDie] = None
    bumped_dice: List[Tuple[DetectedDie, DetectedDie]] = field(default_factory=list)
    pips: int = 0
    multiplier: int = 0
    score: int = 0
    is_miss: bool = False
    displaced_score_delta: int = 0
    total_delta_score: int = 0
    raw_dice_count: int = 0
    previous_dice_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pips": self.pips,
            "multiplier": self.multiplier,
            "score": self.score,
            "is_miss": self.is_miss,
            "displaced_score_delta": self.displaced_score_delta,
            "total_delta_score": self.total_delta_score,
            "raw_dice_count": self.raw_dice_count,
            "previous_dice_count": self.previous_dice_count,
            "new_die": self.new_die.to_dict() if self.new_die else None,
        }


class DeltaDieScorer:
    """
    Maintains baseline resting dice across flicks within each major round.
    Computes spatial and multiplier deltas once motion quiescence is reached,
    isolating the newly flicked die, bump/displacement score adjustments,
    and gutter miss events.
    """

    def __init__(self, match_distance_threshold: float = 75.0) -> None:
        self.match_distance_threshold = float(match_distance_threshold)
        self.previous_dice: List[DetectedDie] = []

    def reset_round(self) -> None:
        """Clears baseline resting dice when a major round starts or board is cleared."""
        self.previous_dice.clear()

    def set_baseline(self, dice: Sequence[DetectedDie]) -> None:
        """Explicitly sets resting dice baseline."""
        self.previous_dice = list(dice)

    def undo_last_flick(self) -> None:
        """Rolls back the last added die from baseline if present."""
        if self.previous_dice:
            self.previous_dice.pop()

    def compute_delta(
        self,
        current_dice: Sequence[DetectedDie],
        active_player_id: Optional[int] = None,
    ) -> DeltaScoringResult:
        """
        Compares currently detected dice against previous resting dice.
        Returns DeltaScoringResult detailing new die, bumps, and flick score.
        """
        valid_current = [d for d in current_dice if not d.is_out_of_bounds and d.zone_multiplier > 0]

        matched_curr_indices = set()
        matched_prev_indices = set()
        bumped_dice: List[Tuple[DetectedDie, DetectedDie]] = []
        displaced_score_delta = 0

        # Build distance pairs between previous dice and valid current dice
        pairs = []
        for p_idx, p_die in enumerate(self.previous_dice):
            for c_idx, c_die in enumerate(valid_current):
                dist = math.hypot(p_die.center[0] - c_die.center[0], p_die.center[1] - c_die.center[1])
                if dist <= self.match_distance_threshold:
                    pairs.append((dist, p_idx, c_idx))

        # Greedy nearest-neighbor matching
        pairs.sort(key=lambda x: x[0])
        for dist, p_idx, c_idx in pairs:
            if p_idx not in matched_prev_indices and c_idx not in matched_curr_indices:
                matched_prev_indices.add(p_idx)
                matched_curr_indices.add(c_idx)
                prev_die = self.previous_dice[p_idx]
                curr_die = valid_current[c_idx]

                # Check if zone or multiplier changed (bumped die)
                if curr_die.zone_multiplier != prev_die.zone_multiplier or dist > 20.0:
                    bumped_dice.append((prev_die, curr_die))
                    displaced_score_delta += (curr_die.score - prev_die.score)

        # Unmatched dice in valid_current are candidate new dice
        unmatched_curr = [
            c_die for c_idx, c_die in enumerate(valid_current)
            if c_idx not in matched_curr_indices
        ]

        # Select new die
        new_die: Optional[DetectedDie] = None
        if unmatched_curr:
            if active_player_id is not None:
                player_matched = [d for d in unmatched_curr if d.player_id == active_player_id]
                if player_matched:
                    new_die = player_matched[0]
                else:
                    new_die = min(unmatched_curr, key=lambda d: getattr(d, "color_distance", 0.0))
            else:
                new_die = unmatched_curr[0]

        if new_die is not None:
            pips = new_die.pip_count
            multiplier = new_die.zone_multiplier
            score = pips * multiplier
            is_miss = False
        else:
            pips = 0
            multiplier = 0
            score = 0
            is_miss = True

        total_delta = score + displaced_score_delta

        res = DeltaScoringResult(
            new_die=new_die,
            bumped_dice=bumped_dice,
            pips=pips,
            multiplier=multiplier,
            score=score,
            is_miss=is_miss,
            displaced_score_delta=displaced_score_delta,
            total_delta_score=total_delta,
            raw_dice_count=len(valid_current),
            previous_dice_count=len(self.previous_dice),
        )

        # Update baseline resting dice
        self.previous_dice = list(valid_current)
        return res


class VisionPipelineThread(QThread):
    """
    Background QThread linking ArUco perspective warp, motion quiescence settle detection,
    and HSV dice detection directly into the live scoreboard HUD.
    
    Supports:
    - Real camera device input (--camera [DEVICE_INDEX]) using Windows DirectShow backend.
    - Interactive software simulation mode (--mock-feed) with animated tumbling dice.
    - Automatic perspective warping via 4 corner ArUco fiducials.
    - Quiescence gating: scoreboard freeze during active motion ('Rolling...'),
      accumulated stillness tracking ('Settling...'), and real-time settle triggers.
    - Delta die scoring isolating active player flicks and bumps.
    """

    frame_ready = pyqtSignal(np.ndarray)       # Processed board frame with HUD annotations
    raw_frame_ready = pyqtSignal(np.ndarray)   # Raw unrectified camera frame
    motion_status = pyqtSignal(object)         # MotionResult object
    settle_event = pyqtSignal(object)          # DeltaScoringResult object
    calibration_status = pyqtSignal(bool)      # True if ArUco board homography is locked
    error_occurred = pyqtSignal(str)          # Error notification string

    def __init__(
        self,
        camera_idx: Optional[int] = None,
        mock_feed: bool = False,
        config_path: str = DEFAULT_CONFIG_PATH,
        quiescence_duration: float = 1.2,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.camera_idx = camera_idx
        self.mock_feed = mock_feed
        self.config_path = config_path
        self.quiescence_duration = float(quiescence_duration)

        self._running = False
        self._paused = False
        self._active_player_id: int = 1
        self._active_player_color: str = "#EF4444"

        # Dimensions & Zone Configuration
        self.board_width = DEFAULT_BOARD_WIDTH
        self.board_height = DEFAULT_BOARD_HEIGHT
        self.zones: List[ScoringZone] = []
        self._load_zones()

        # Calibration (ArUco fiducials)
        self.calibrator = ArucoCalibrator(
            board_width=self.board_width,
            board_height=self.board_height,
        )

        # Motion detector with quiescence gating
        self.motion_detector = MotionDetector(
            config=MotionConfig(
                quiescence_duration=self.quiescence_duration,
                motion_threshold=400.0,
            )
        )

        # Player profile manager and zone scorer
        self.player_manager = PlayerProfileManager.create_default_roster(10)
        self.zone_scorer = ZoneScorer(zones_source=self.zones)
        self.detection_pipeline = DetectionPipeline(
            player_manager=self.player_manager,
            zone_scorer=self.zone_scorer,
        )

        # Delta die scoring engine
        self.delta_scorer = DeltaDieScorer()

        # Mock simulation state
        self._mock_sheet = generate_aruco_calibration_sheet(self.board_width, self.board_height)
        self._mock_placed_dice: List[Dict[str, Any]] = []
        self._mock_rolling = False
        self._mock_roll_start_time = 0.0
        self._mock_roll_duration = 0.7
        self._mock_current_roll: Optional[Dict[str, Any]] = None
        self._mock_last_settle_time = time.time()
        self.mock_auto_running = False
        self._last_warped: Optional[np.ndarray] = None

    def _load_zones(self) -> None:
        """Loads zones from configuration file or applies default fallback tiered zones."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                dims = data.get("board_dimensions", {})
                self.board_width = dims.get("width", DEFAULT_BOARD_WIDTH)
                self.board_height = dims.get("height", DEFAULT_BOARD_HEIGHT)
                for z_data in data.get("zones", []):
                    try:
                        self.zones.append(ScoringZone.from_dict(z_data))
                    except Exception:
                        pass
                if self.zones:
                    return
            except Exception as e:
                logger.warning("Could not load zones from %s: %s", self.config_path, e)

        # Fallback tiered zones
        self.zones = [
            ScoringZone("Zone 1x", 1, [(100, 200), (700, 200), (700, 400), (100, 400)], (255, 140, 0)),
            ScoringZone("Zone 2x", 2, [(100, 400), (700, 400), (700, 650), (100, 650)], (50, 205, 50)),
            ScoringZone("Zone 3x", 3, [(100, 650), (700, 650), (700, 900), (100, 900)], (30, 30, 220)),
            ScoringZone("Zone 4x", 4, [(100, 900), (700, 900), (700, 1100), (100, 1100)], (0, 165, 255)),
        ]

    def set_active_player(self, player_id: int, color_hex: str) -> None:
        """Updates active player context for color matching and delta scoring."""
        self._active_player_id = int(player_id)
        self._active_player_color = str(color_hex)

    def reset_round(self) -> None:
        """Resets baseline dice tracking and clears mock board for a new major round."""
        self.delta_scorer.reset_round()
        self._mock_placed_dice.clear()
        self._mock_rolling = False
        self._mock_current_roll = None
        self._mock_last_settle_time = time.time()

    def undo_flick(self) -> None:
        """Rolls back the last flick from delta scoring baseline and mock board."""
        self.delta_scorer.undo_last_flick()
        if self._mock_placed_dice:
            self._mock_placed_dice.pop()

    def trigger_mock_flick(self) -> None:
        """Triggers a simulated dice roll event in software simulation mode."""
        if self._mock_rolling:
            return

        import random
        prof = self.player_manager.get_profile(self._active_player_id)
        bg_color = prof.display_bgr if prof else (30, 30, 220)

        # 12% gutter miss chance
        is_miss = (random.random() < 0.12)
        if is_miss:
            target_cx = random.choice([30, 770])
            target_cy = random.randint(1100, 1180)
            pips = random.randint(1, 6)
        else:
            target_mult = random.choice([1, 2, 3, 4])
            target_cy = 280 + target_mult * 200 + random.randint(-40, 40)
            target_cx = random.randint(220, 580)
            pips = random.randint(1, 6)

        self._mock_current_roll = {
            "cx": target_cx,
            "cy": target_cy,
            "size": 60,
            "pips": pips,
            "bg_color": bg_color,
            "player_id": self._active_player_id,
            "is_miss": is_miss,
            "start_cx": random.randint(320, 480),
            "start_cy": 130,
        }
        self._mock_roll_start_time = time.time()
        self._mock_rolling = True

    def add_mock_die(self, pips: int, multiplier: int, x: float, y: float, player_id: int) -> None:
        """Directly adds a placed die to the mock simulation scene."""
        prof = self.player_manager.get_profile(player_id)
        bg = prof.display_bgr if prof else (30, 30, 220)
        self._mock_placed_dice.append({
            "cx": int(x),
            "cy": int(y),
            "size": 60,
            "pips": int(pips),
            "bg_color": bg,
            "player_id": int(player_id),
            "is_miss": False,
        })

    def force_settle(self) -> None:
        """Forces an immediate settle detection pass on current frame."""
        if self._last_warped is not None:
            warped = self._last_warped
        else:
            warped = np.full((self.board_height, self.board_width, 3), 200, dtype=np.uint8)
        scoring_res, _ = self.detection_pipeline.process_frame(warped)
        delta = self.delta_scorer.compute_delta(
            scoring_res.dice,
            active_player_id=self._active_player_id,
        )
        self.settle_event.emit(delta)

    def pause(self) -> None:
        """Pauses frame processing loop."""
        self._paused = True

    def resume(self) -> None:
        """Resumes frame processing loop."""
        self._paused = False

    def stop(self, timeout_ms: int = 2000) -> None:
        """Stops the thread gracefully and waits for termination."""
        self._running = False
        self.wait(timeout_ms)

    def start(self, priority: QThread.Priority = QThread.Priority.InheritPriority) -> None:
        """Starts background worker thread."""
        self._running = True
        super().start(priority)

    def run(self) -> None:
        """Main execution loop."""
        if not self._running:
            return
        if self.mock_feed or self.camera_idx is None:
            self._run_mock_loop()
        else:
            self._run_camera_loop()

    def _run_camera_loop(self) -> None:
        """Captures frames from physical camera device and processes pipeline."""
        camera_idx = self.camera_idx if self.camera_idx is not None else 0
        cap = None
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                if cap is not None:
                    cap.release()
                cap = cv2.VideoCapture(camera_idx)
        else:
            cap = cv2.VideoCapture(camera_idx)

        if not cap.isOpened():
            err = f"Could not open camera device index {camera_idx}"
            logger.error(err)
            self.error_occurred.emit(err)
            self._running = False
            return

        try:
            while self._running:
                if self._paused:
                    time.sleep(0.05)
                    continue

                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    time.sleep(0.02)
                    continue

                self._process_single_frame(frame)
                time.sleep(0.03)  # ~30 FPS
        finally:
            if cap is not None:
                cap.release()

    def _run_mock_loop(self) -> None:
        """Generates synthetic video feed with ArUco markers, tumbling dice, and settle events."""
        while self._running:
            if self._paused:
                time.sleep(0.05)
                continue

            now = time.time()

            # Handle auto roll interval
            if self.mock_auto_running and not self._mock_rolling:
                if (now - self._mock_last_settle_time) > 2.8:
                    self.trigger_mock_flick()

            # Build current frame dice specs
            current_dice_specs = list(self._mock_placed_dice)

            if self._mock_rolling and self._mock_current_roll:
                elapsed = now - self._mock_roll_start_time
                progress = min(1.0, elapsed / self._mock_roll_duration)

                if progress < 1.0:
                    s_cx = self._mock_current_roll["start_cx"]
                    s_cy = self._mock_current_roll["start_cy"]
                    e_cx = self._mock_current_roll["cx"]
                    e_cy = self._mock_current_roll["cy"]
                    wobble = math.sin(progress * 12.0) * 20.0
                    curr_cx = int(s_cx + progress * (e_cx - s_cx) + wobble)
                    curr_cy = int(s_cy + progress * (e_cy - s_cy))
                    angle = progress * 720.0

                    rolling_spec = dict(self._mock_current_roll)
                    rolling_spec["cx"] = curr_cx
                    rolling_spec["cy"] = curr_cy
                    rolling_spec["angle"] = angle
                    current_dice_specs.append(rolling_spec)
                else:
                    self._mock_rolling = False
                    if not self._mock_current_roll.get("is_miss", False):
                        self._mock_placed_dice.append(dict(self._mock_current_roll))

            # Render synthetic board image
            board_img, _ = create_synthetic_board(
                width=self.board_width,
                height=self.board_height,
                dice_specs=current_dice_specs,
                zones=self.zones,
            )

            # Stamp ArUco fiducials at the corners
            sheet = self._mock_sheet
            board_img[0:180, 0:180] = sheet[0:180, 0:180]
            board_img[0:180, self.board_width - 180 : self.board_width] = sheet[0:180, self.board_width - 180 : self.board_width]
            board_img[self.board_height - 180 : self.board_height, self.board_width - 180 : self.board_width] = sheet[self.board_height - 180 : self.board_height, self.board_width - 180 : self.board_width]
            board_img[self.board_height - 180 : self.board_height, 0:180] = sheet[self.board_height - 180 : self.board_height, 0:180]

            self._process_single_frame(board_img, timestamp=now)
            time.sleep(0.035)  # ~28 FPS

    def _process_single_frame(self, raw_frame: np.ndarray, timestamp: Optional[float] = None) -> None:
        """Processes a single camera/synthetic frame through the complete multi-stage pipeline."""
        now = timestamp if timestamp is not None else time.time()
        self.raw_frame_ready.emit(raw_frame)

        # 1. ArUco Marker Detection & Perspective Homography Warp
        if self.calibrator.homography_matrix is None:
            calib_success = self.calibrator.calibrate_from_frame(raw_frame)
            self.calibration_status.emit(calib_success)

        if self.calibrator.homography_matrix is not None:
            try:
                warped = self.calibrator.warp_frame(raw_frame)
            except Exception:
                warped = cv2.resize(raw_frame, (self.board_width, self.board_height))
        else:
            warped = cv2.resize(raw_frame, (self.board_width, self.board_height))

        self._last_warped = warped

        # 2. Motion Quiescence Gating & Settle Detection
        motion_res = self.motion_detector.process_frame(warped, timestamp=now)
        self.motion_status.emit(motion_res)

        # 3. Real-Time Settle Trigger & Delta Die Scoring
        if motion_res.just_settled:
            self._mock_last_settle_time = now
            scoring_res, _ = self.detection_pipeline.process_frame(warped)
            delta = self.delta_scorer.compute_delta(
                scoring_res.dice,
                active_player_id=self._active_player_id,
            )
            self.settle_event.emit(delta)

        # 4. Render Live HUD Broadcast Frame
        annotated = self._render_live_hud(warped, motion_res)
        self.frame_ready.emit(annotated)

    def _render_live_hud(self, warped: np.ndarray, motion_res: MotionResult) -> np.ndarray:
        """Renders live broadcast HUD overlays directly on the rectified board image."""
        output = warped.copy()

        # 1. Overlay scoring zones
        if self.zone_scorer.zone_manager:
            output = self.zone_scorer.zone_manager.render_overlay(output, alpha=0.22)

        # 2. Overlay detected dice badges from last snapshot
        if self.detection_pipeline.current_result:
            for d in self.detection_pipeline.current_result.dice:
                if d.zone_multiplier > 0 and not d.is_out_of_bounds:
                    color = d.display_bgr if d.player_id else (255, 255, 255)
                    cx, cy = int(d.center[0]), int(d.center[1])
                    cv2.circle(output, (cx, cy), 30, color, 3, cv2.LINE_AA)
                    cv2.circle(output, (cx, cy), 3, (255, 255, 255), -1, cv2.LINE_AA)
                    pname = d.player_name or f"Player {d.player_id}"
                    badge = f"{pname}: {d.pip_count}x{d.zone_multiplier}={d.score}pts"
                    cv2.putText(output, badge, (max(10, cx - 60), max(25, cy - 36)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(output, badge, (max(10, cx - 60), max(25, cy - 36)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Motion status overlay and quiescence progress bar
        output = self.motion_detector.draw_overlay(output, motion_res)

        # 4. ArUco homography lock badge
        is_cal = (self.calibrator.homography_matrix is not None)
        cal_text = "ArUco: Calibrated & Warped" if is_cal else "ArUco: Scanning for fiducials..."
        cal_col = (50, 205, 50) if is_cal else (0, 165, 255)
        cv2.putText(output, cal_text, (self.board_width - 290, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, cal_col, 1, cv2.LINE_AA)

        return output


# ============================================================================
# Surface-Optimized UI Widgets
# ============================================================================

class TouchButton(QPushButton):
    """Touch-optimized button with minimum tactile sizing and haptic styling."""

    def __init__(
        self,
        text: str = "",
        min_height: int = 48,
        min_width: int = 48,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(text, parent)
        self.setMinimumHeight(min_height)
        self.setMinimumWidth(min_width)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)


class TurnBannerWidget(QFrame):
    """
    Hero Banner displayed prominently at the top of the HUD.
    Displays:
    - Current Major Round (1/4)
    - Snake Order Direction (Forward 1➔N or Reverse N➔1)
    - Active Player badge, name, and color swatch
    - Flick progress within the round (Flick 2 of 4)
    - Live round score and cumulative total score
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("turnBanner")
        self.setProperty("class", "cardFrame")
        self.setMinimumHeight(110)
        self.init_ui()

    def init_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 12, 18, 12)
        layout.setSpacing(20)

        # Left Section: Round & Snake Order
        left_layout = QVBoxLayout()
        left_layout.setSpacing(4)

        self.lbl_round = QLabel("ROUND 1 OF 4")
        self.lbl_round.setStyleSheet("font-size: 20px; font-weight: 800; color: #38BDF8;")
        left_layout.addWidget(self.lbl_round)

        self.lbl_direction = QLabel("Order: Forward (1 ➔ N)")
        self.lbl_direction.setStyleSheet("font-size: 15px; font-weight: 600; color: #94A3B8;")
        left_layout.addWidget(self.lbl_direction)

        self.lbl_turn_counter = QLabel("Turn 1 of 16")
        self.lbl_turn_counter.setStyleSheet("font-size: 13px; color: #64748B;")
        left_layout.addWidget(self.lbl_turn_counter)

        layout.addLayout(left_layout, stretch=1)

        # Center Section: Active Player Hero Card
        center_layout = QHBoxLayout()
        center_layout.setSpacing(14)

        self.color_pill = QFrame()
        self.color_pill.setFixedSize(32, 70)
        self.color_pill.setStyleSheet("border-radius: 8px; background-color: #EF4444;")
        center_layout.addWidget(self.color_pill)

        player_info_layout = QVBoxLayout()
        player_info_layout.setSpacing(2)

        self.lbl_turn_badge = QLabel("ACTIVE FLICKER")
        self.lbl_turn_badge.setStyleSheet(
            "font-size: 12px; font-weight: 800; color: #F59E0B; letter-spacing: 1.5px;"
        )
        player_info_layout.addWidget(self.lbl_turn_badge)

        self.lbl_player_name = QLabel("Player 1")
        self.lbl_player_name.setStyleSheet("font-size: 28px; font-weight: 900; color: #FFFFFF;")
        player_info_layout.addWidget(self.lbl_player_name)

        center_layout.addLayout(player_info_layout)
        layout.addLayout(center_layout, stretch=2)

        # Right Section: Flick Progress & Live Score
        right_layout = QVBoxLayout()
        right_layout.setSpacing(4)
        right_layout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.lbl_flick_progress = QLabel("Flick 1 of 4")
        self.lbl_flick_progress.setStyleSheet("font-size: 19px; font-weight: 700; color: #F8FAFC;")
        right_layout.addWidget(self.lbl_flick_progress, alignment=Qt.AlignmentFlag.AlignRight)

        self.lbl_score_summary = QLabel("Round: 0 pts | Total: 0 pts")
        self.lbl_score_summary.setStyleSheet("font-size: 16px; font-weight: 600; color: #34D399;")
        right_layout.addWidget(self.lbl_score_summary, alignment=Qt.AlignmentFlag.AlignRight)

        # 4 Flick indicator dots
        self.dots_layout = QHBoxLayout()
        self.dots_layout.setSpacing(6)
        self.dot_widgets: List[QFrame] = []
        for _ in range(FLICKS_PER_PLAYER_PER_ROUND):
            dot = QFrame()
            dot.setFixedSize(14, 14)
            dot.setStyleSheet("border-radius: 7px; background-color: #475569;")
            self.dots_layout.addWidget(dot)
            self.dot_widgets.append(dot)
        right_layout.addLayout(self.dots_layout)

        layout.addLayout(right_layout, stretch=1)

    def update_view(self, state: GameState) -> None:
        """Refreshes the hero banner to reflect current game state."""
        if state.is_game_over:
            self.lbl_round.setText("GAME COMPLETE")
            self.lbl_round.setStyleSheet("font-size: 20px; font-weight: 800; color: #F59E0B;")
            self.lbl_direction.setText("All 4 Major Rounds Finished")
            self.lbl_turn_badge.setText("CHAMPION")
            winners = state.get_winners()
            winner_names = ", ".join(w.name for w in winners)
            self.lbl_player_name.setText(winner_names if winner_names else "Game Over")
            self.lbl_flick_progress.setText("Final Standings")
            if winners:
                self.color_pill.setStyleSheet(f"border-radius: 8px; background-color: {winners[0].color};")
                self.lbl_score_summary.setText(f"Winning Score: {winners[0].total_score()} pts")
            for dot in self.dot_widgets:
                dot.setStyleSheet("border-radius: 7px; background-color: #10B981;")
            return

        # Round status
        round_num = state.current_round
        self.lbl_round.setText(f"ROUND {round_num} OF {state.total_rounds}")
        self.lbl_round.setStyleSheet("font-size: 20px; font-weight: 800; color: #38BDF8;")
        direction_name = state.get_round_direction_name(round_num)
        self.lbl_direction.setText(f"Order: {direction_name}")
        self.lbl_turn_counter.setText(
            f"Turn {state.current_turn_index + 1} of {state.total_turns_per_round}"
        )

        turn = state.get_current_turn()
        if turn:
            player, flick_num = turn
            self.lbl_turn_badge.setText("ACTIVE FLICKER")
            self.lbl_player_name.setText(player.name)
            self.color_pill.setStyleSheet(f"border-radius: 8px; background-color: {player.color};")
            self.lbl_flick_progress.setText(f"Flick {flick_num} of {state.flicks_per_player}")
            self.lbl_score_summary.setText(
                f"Round: {player.round_score(round_num)} pts | Total: {player.total_score()} pts"
            )

            # Update flick dots (flick_num is 1..4)
            for i, dot in enumerate(self.dot_widgets):
                if i < flick_num - 1:
                    dot.setStyleSheet(f"border-radius: 7px; background-color: {player.color};")
                elif i == flick_num - 1:
                    dot.setStyleSheet("border-radius: 7px; background-color: #F59E0B; border: 2px solid #FFFFFF;")
                else:
                    dot.setStyleSheet("border-radius: 7px; background-color: #475569;")
        elif state.is_current_round_complete():
            self.lbl_turn_badge.setText("ROUND FINISHED")
            self.lbl_player_name.setText("Ready for Next Round")
            self.lbl_flick_progress.setText("Clear Board")
            self.lbl_score_summary.setText("Scores Locked")
            for dot in self.dot_widgets:
                dot.setStyleSheet("border-radius: 7px; background-color: #10B981;")


class LeaderboardWidget(QFrame):
    """
    Live Sidebar Leaderboard.
    Displays players sorted by cumulative total score, with breakdowns for
    Rounds 1, 2, 3, 4, current tentative round score, and total score.
    Highlights active player row with high contrast indicator.
    """

    cell_edited = pyqtSignal(int, int, int)  # player_id, round_num, new_score

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("leaderboardWidget")
        self.setProperty("class", "cardFrame")
        self.init_ui()

    def init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header_layout = QHBoxLayout()
        title = QLabel("LIVE LEADERBOARD")
        title.setStyleSheet("font-size: 17px; font-weight: 800; color: #F8FAFC; letter-spacing: 1px;")
        header_layout.addWidget(title)

        header_layout.addStretch()
        self.lbl_player_count = QLabel("4 Players")
        self.lbl_player_count.setStyleSheet("font-size: 13px; color: #94A3B8;")
        header_layout.addWidget(self.lbl_player_count)
        layout.addLayout(header_layout)

        # Leaderboard Table
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "Rank", "Player", "R1", "R2", "R3", "R4", "Total"
        ])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Touch-friendly row height
        self.table.verticalHeader().setDefaultSectionSize(44)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)  # Player name expands
        self.table.setColumnWidth(0, 56)   # Rank
        self.table.setColumnWidth(2, 42)   # R1
        self.table.setColumnWidth(3, 42)   # R2
        self.table.setColumnWidth(4, 42)   # R3
        self.table.setColumnWidth(5, 42)   # R4
        self.table.setColumnWidth(6, 68)   # Total

        layout.addWidget(self.table)

    def update_view(self, state: GameState) -> None:
        """Re-sorts and updates table rows based on current GameState."""
        self.lbl_player_count.setText(f"{state.player_count} Players")
        leaderboard = state.get_leaderboard()
        active_player = state.get_active_player()

        self.table.setRowCount(len(leaderboard))
        for row, player in enumerate(leaderboard):
            # Rank medal / number
            rank_text = f"#{row + 1}"
            if row == 0:
                rank_text = "🥇 1"
            elif row == 1:
                rank_text = "🥈 2"
            elif row == 2:
                rank_text = "🥉 3"

            item_rank = QTableWidgetItem(rank_text)
            item_rank.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            # Player Name & color indicator
            item_player = QTableWidgetItem(f"● {player.name}")
            item_player.setForeground(QBrush(QColor(player.color)))
            font = item_player.font()
            font.setBold(True)
            item_player.setFont(font)

            # Round breakdown
            r1 = QTableWidgetItem(str(player.round_score(1)))
            r1.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            r2 = QTableWidgetItem(str(player.round_score(2)))
            r2.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            r3 = QTableWidgetItem(str(player.round_score(3)))
            r3.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            r4 = QTableWidgetItem(str(player.round_score(4)))
            r4.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            # Total
            total = QTableWidgetItem(str(player.total_score()))
            total.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            total_font = total.font()
            total_font.setBold(True)
            total.setFont(total_font)
            total.setForeground(QBrush(QColor("#4ADE80")))

            # Check if active player
            is_active = (active_player is not None and player.id == active_player.id and not state.is_game_over)
            row_items = [item_rank, item_player, r1, r2, r3, r4, total]
            for col, item in enumerate(row_items):
                if is_active:
                    item.setBackground(QBrush(QColor("#1E3A8A")))
                self.table.setItem(row, col, item)


class TouchScoreWidget(QFrame):
    """
    Surface Touch-Optimized Input Controls.
    Includes:
    - Giant Spacebar / "0 / MISS (SPACE)" button (minimum height 74px)
    - Touch Keypad for Pip Count (1..6)
    - Touch Keypad for Multipliers (1x..6x)
    - Live Score Preview & Big Green Confirm Button (minimum height 60px)
    - Large Undo & Round Action buttons
    """

    score_confirmed = pyqtSignal(int, int, int)  # score, pips, multiplier
    miss_triggered = pyqtSignal()
    undo_triggered = pyqtSignal()
    next_round_triggered = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("touchScoreWidget")
        self.setProperty("class", "cardFrame")
        self.selected_pip: int = 1
        self.selected_multiplier: int = 1
        self.init_ui()

    def init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)

        # 1. Sleek Prominent Miss Button (Spacebar Override)
        self.btn_miss = TouchButton(
            "0 / MISS  [SPACE]",
            min_height=42,
            parent=self,
        )
        self.btn_miss.setObjectName("missButton")
        self.btn_miss.setStyleSheet(
            "background-color: #DC2626; color: #FFFFFF; font-size: 15px; font-weight: 800; "
            "border-radius: 8px; border: 1px solid #EF4444;"
        )
        self.btn_miss.setToolTip("Record 0 points (die off board / gutter) and advance turn [Space]")
        self.btn_miss.clicked.connect(self._on_miss_clicked)
        layout.addWidget(self.btn_miss)

        # 2. Compact Combined Selector Rows: Pips & Multipliers
        selector_grid = QGridLayout()
        selector_grid.setContentsMargins(0, 0, 0, 0)
        selector_grid.setHorizontalSpacing(8)
        selector_grid.setVerticalSpacing(4)

        lbl_pips = QLabel("PIPS:")
        lbl_pips.setStyleSheet("font-size: 12px; font-weight: 800; color: #94A3B8; min-width: 45px;")
        selector_grid.addWidget(lbl_pips, 0, 0)

        pip_row = QHBoxLayout()
        pip_row.setSpacing(6)
        self.pip_buttons: List[QPushButton] = []
        for i in range(1, 7):
            btn = TouchButton(str(i), min_height=36, min_width=38)
            btn.setProperty("class", "gridSelectorBtn")
            btn.setStyleSheet("font-size: 14px; font-weight: 700;")
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, val=i: self.select_pip(val))
            pip_row.addWidget(btn)
            self.pip_buttons.append(btn)
        selector_grid.addLayout(pip_row, 0, 1)

        lbl_mult = QLabel("ZONE:")
        lbl_mult.setStyleSheet("font-size: 12px; font-weight: 800; color: #94A3B8; min-width: 45px;")
        selector_grid.addWidget(lbl_mult, 1, 0)

        mult_row = QHBoxLayout()
        mult_row.setSpacing(6)
        self.mult_buttons: List[QPushButton] = []
        for i in range(1, 7):
            btn = TouchButton(f"{i}x", min_height=36, min_width=38)
            btn.setProperty("class", "gridSelectorBtn")
            btn.setStyleSheet("font-size: 14px; font-weight: 700;")
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, val=i: self.select_multiplier(val))
            mult_row.addWidget(btn)
            self.mult_buttons.append(btn)
        selector_grid.addLayout(mult_row, 1, 1)

        layout.addLayout(selector_grid)

        # 3. Action Bar: Confirm & Undo in clean single row
        action_row = QHBoxLayout()
        action_row.setSpacing(8)

        self.btn_confirm = TouchButton(
            "✓ CONFIRM: 1 PT [ENTER]",
            min_height=42,
            parent=self,
        )
        self.btn_confirm.setObjectName("confirmButton")
        self.btn_confirm.setStyleSheet(
            "background-color: #10B981; color: #042F2E; font-size: 14px; font-weight: 800; "
            "border-radius: 8px; border: 1px solid #34D399;"
        )
        self.btn_confirm.clicked.connect(self._on_confirm_clicked)
        action_row.addWidget(self.btn_confirm, stretch=3)

        self.btn_undo = TouchButton("↩ Undo [Ctrl+Z]", min_height=42, min_width=110)
        self.btn_undo.setStyleSheet("background-color: #475569; font-size: 13px; font-weight: 600; border-radius: 8px;")
        self.btn_undo.clicked.connect(self.undo_triggered.emit)
        action_row.addWidget(self.btn_undo, stretch=1)

        self.btn_next_round = TouchButton("Next Round ➔", min_height=42, min_width=120)
        self.btn_next_round.setStyleSheet("background-color: #2563EB; font-size: 14px; font-weight: bold; border-radius: 8px;")
        self.btn_next_round.clicked.connect(self.next_round_triggered.emit)
        self.btn_next_round.setVisible(False)
        action_row.addWidget(self.btn_next_round, stretch=1)

        layout.addLayout(action_row)

        # Set default selections
        self.select_pip(1)
        self.select_multiplier(1)

    def select_pip(self, val: int) -> None:
        """Selects pip face value (1..6) and updates button states."""
        self.selected_pip = val
        for i, btn in enumerate(self.pip_buttons, start=1):
            btn.setChecked(i == val)
        self._update_calculation_preview()

    def select_multiplier(self, val: int) -> None:
        """Selects zone multiplier (1..6) and updates button states."""
        self.selected_multiplier = val
        for i, btn in enumerate(self.mult_buttons, start=1):
            btn.setChecked(i == val)
        self._update_calculation_preview()

    def _update_calculation_preview(self) -> None:
        pts = self.selected_pip * self.selected_multiplier
        if hasattr(self, "lbl_calc_preview"):
            self.lbl_calc_preview.setText(
                f"Calculation: {self.selected_pip} pips × {self.selected_multiplier}x = {pts} pts"
            )
        self.btn_confirm.setText(f"✓ CONFIRM: {pts} PTS [ENTER]")

    def _on_confirm_clicked(self) -> None:
        score = self.selected_pip * self.selected_multiplier
        self.score_confirmed.emit(score, self.selected_pip, self.selected_multiplier)

    def _on_miss_clicked(self) -> None:
        self.miss_triggered.emit()

    def set_round_complete_mode(self, complete: bool, next_round_num: int = 0) -> None:
        """Toggles visibility of the Next Round button when round completes."""
        self.btn_next_round.setVisible(complete)
        if complete:
            self.btn_next_round.setText(f"Start Round {next_round_num} ➔")


class BoardOverlayWidget(QFrame):
    """
    Visual Board & Scoring Zones representation.
    Renders board layout from zones_config.json, with color-coded polygons,
    multiplier badges, and active placed dice markers.
    Tapping a zone directly selects that zone multiplier in TouchScoreWidget.
    """

    zone_selected = pyqtSignal(int)  # multiplier (1..6)

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("boardOverlayWidget")
        self.setProperty("class", "cardFrame")
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.config_path = config_path
        self.board_width = 800
        self.board_height = 1200
        self.zones: List[Dict[str, Any]] = []
        self.active_dice: List[Dict[str, Any]] = []
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_qimage: Optional[QImage] = None
        self.load_zones()

    def set_camera_frame(self, frame: np.ndarray) -> None:
        """Updates the live camera video frame displayed on the board."""
        if frame is None or frame.size == 0:
            return
        self.latest_frame = frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        self.latest_qimage = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        self.update()

    def load_zones(self) -> None:
        """Loads scoring zones and board dimensions from zones_config.json if available."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                dims = data.get("board_dimensions", {})
                self.board_width = dims.get("width", 800)
                self.board_height = dims.get("height", 1200)
                self.zones = data.get("zones", [])
                return
            except Exception as e:
                logger.warning("Failed to load zones_config.json: %s", e)

        # Fallback synthetic tiered zones
        self.zones = [
            {"name": "Zone 1x", "multiplier": 1, "points": [[100, 200], [700, 200], [700, 400], [100, 400]], "color": [0, 140, 255]},
            {"name": "Zone 2x", "multiplier": 2, "points": [[100, 400], [700, 400], [700, 650], [100, 650]], "color": [50, 205, 50]},
            {"name": "Zone 3x", "multiplier": 3, "points": [[100, 650], [700, 650], [700, 900], [100, 900]], "color": [220, 30, 30]},
            {"name": "Zone 4x", "multiplier": 4, "points": [[100, 900], [700, 900], [700, 1100], [100, 1100]], "color": [255, 165, 0]},
        ]

    def add_die_pin(self, x: float, y: float, player_color: str, pips: int, multiplier: int) -> None:
        """Adds a visual die pin to the board overlay."""
        self.active_dice.append({
            "x": x,
            "y": y,
            "color": player_color,
            "pips": pips,
            "multiplier": multiplier,
            "score": pips * multiplier,
        })
        self.update()

    def clear_dice(self) -> None:
        """Clears all active die pins (at end of round)."""
        self.active_dice.clear()
        self.update()

    def _board_to_widget_coords(self, bx: float, by: float) -> Tuple[float, float]:
        """Maps board (0..board_width, 0..board_height) to widget pixel coordinates."""
        w, h = self.width() - 24, self.height() - 24
        scale_x = w / self.board_width
        scale_y = h / self.board_height
        scale = min(scale_x, scale_y)
        offset_x = 12 + (w - self.board_width * scale) / 2
        offset_y = 12 + (h - self.board_height * scale) / 2
        return offset_x + bx * scale, offset_y + by * scale

    def _widget_to_board_coords(self, wx: float, wy: float) -> Tuple[float, float]:
        """Maps widget pixel coordinates to board coordinates."""
        w, h = self.width() - 24, self.height() - 24
        scale_x = w / self.board_width
        scale_y = h / self.board_height
        scale = min(scale_x, scale_y)
        offset_x = 12 + (w - self.board_width * scale) / 2
        offset_y = 12 + (h - self.board_height * scale) / 2
        if scale <= 0:
            return 0.0, 0.0
        bx = (wx - offset_x) / scale
        by = (wy - offset_y) / scale
        return bx, by

    def paintEvent(self, event: Any) -> None:
        """Paints the board background, polygon zones, and placed dice pins or live camera feed."""
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw board border
        top_left_x, top_left_y = self._board_to_widget_coords(0, 0)
        bot_right_x, bot_right_y = self._board_to_widget_coords(self.board_width, self.board_height)
        board_rect = QRectF(top_left_x, top_left_y, bot_right_x - top_left_x, bot_right_y - top_left_y)

        if self.latest_qimage is not None and not self.latest_qimage.isNull():
            # Draw live camera / warped video feed
            painter.drawImage(board_rect, self.latest_qimage)
            painter.setPen(QPen(QColor("#38BDF8"), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(board_rect)
        else:
            painter.fillRect(board_rect, QColor("#111827"))
            painter.setPen(QPen(QColor("#38BDF8"), 2))
            painter.drawRect(board_rect)

            # Draw Zones
            for zone in self.zones:
                pts = zone.get("points", [])
                if len(pts) < 3:
                    continue
                poly = QPolygonF()
                centroid_x, centroid_y = 0.0, 0.0
                for pt in pts:
                    wx, wy = self._board_to_widget_coords(pt[0], pt[1])
                    poly.append(QPointF(wx, wy))
                    centroid_x += wx
                    centroid_y += wy
                centroid_x /= len(pts)
                centroid_y /= len(pts)

                raw_col = zone.get("color", [0, 180, 255])
                # Handle RGB vs BGR safely
                fill_col = QColor(raw_col[0], raw_col[1], raw_col[2], 75)
                stroke_col = QColor(raw_col[0], raw_col[1], raw_col[2], 220)

                painter.setBrush(QBrush(fill_col))
                painter.setPen(QPen(stroke_col, 2))
                painter.drawPolygon(poly)

                # Zone label
                mult = zone.get("multiplier", 1)
                painter.setPen(QPen(QColor("#FFFFFF")))
                painter.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
                painter.drawText(
                    QRectF(centroid_x - 30, centroid_y - 15, 60, 30),
                    Qt.AlignmentFlag.AlignCenter,
                    f"{mult}x",
                )

            # Draw Placed Dice
            for die in self.active_dice:
                wx, wy = self._board_to_widget_coords(die["x"], die["y"])
                die_rect = QRectF(wx - 14, wy - 14, 28, 28)
                painter.setBrush(QBrush(QColor(die["color"])))
                painter.setPen(QPen(QColor("#FFFFFF"), 2))
                painter.drawRoundedRect(die_rect, 6, 6)

                # Draw pip text
                painter.setPen(QPen(QColor("#000000") if die["color"] == "#E2E8F0" else QColor("#FFFFFF")))
                painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
                painter.drawText(die_rect, Qt.AlignmentFlag.AlignCenter, str(die["pips"]))

        painter.end()

    def mousePressEvent(self, event: Any) -> None:
        """Handles touch or click on the board to select corresponding zone multiplier."""
        wx = event.position().x()
        wy = event.position().y()
        bx, by = self._widget_to_board_coords(wx, wy)

        # Point in polygon check for each zone
        for zone in self.zones:
            pts = zone.get("points", [])
            if self._is_point_in_polygon(bx, by, pts):
                mult = zone.get("multiplier", 1)
                self.zone_selected.emit(mult)
                break
        super().mousePressEvent(event)

    @staticmethod
    def _is_point_in_polygon(x: float, y: float, poly: Sequence[Sequence[float]]) -> bool:
        """Pure Python Ray-casting point-in-polygon algorithm."""
        n = len(poly)
        if n < 3:
            return False
        inside = False
        p1x, p1y = poly[0][0], poly[0][1]
        for i in range(1, n + 1):
            p2x, p2y = poly[i % n][0], poly[i % n][1]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside


class PlayerSetupDialog(QDialog):
    """Touch-friendly dialog to configure player names, count (1..10), and colors."""

    def __init__(self, current_players: List[Player], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Configure Players (Up to 10)")
        self.setMinimumSize(480, 520)
        self.setStyleSheet(SURFACE_QSS)
        self.players = [Player(p.id, p.name, p.color) for p in current_players]
        self.init_ui()

    def init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        header = QLabel("Tumblin' Dice Player Roster")
        header.setStyleSheet("font-size: 20px; font-weight: bold; color: #38BDF8;")
        layout.addWidget(header)

        # Player rows scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        self.rows_layout = QVBoxLayout(container)
        self.rows_layout.setSpacing(10)

        self.name_inputs: List[QLineEdit] = []
        for i, player in enumerate(self.players):
            row = QHBoxLayout()
            color_box = QFrame()
            color_box.setFixedSize(28, 44)
            color_box.setStyleSheet(f"border-radius: 6px; background-color: {player.color};")
            row.addWidget(color_box)

            edit = QLineEdit(player.name)
            edit.setMinimumHeight(44)
            edit.setStyleSheet("font-size: 16px; padding-left: 8px;")
            row.addWidget(edit, stretch=1)
            self.name_inputs.append(edit)
            self.rows_layout.addLayout(row)

        scroll.setWidget(container)
        layout.addWidget(scroll)

        # Dialog Buttons
        btn_box = QHBoxLayout()
        btn_box.setSpacing(12)

        btn_cancel = TouchButton("Cancel", min_height=50)
        btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(btn_cancel)

        btn_save = TouchButton("Save & Start Match", min_height=50)
        btn_save.setStyleSheet("background-color: #16A34A; color: white;")
        btn_save.clicked.connect(self._on_save)
        btn_box.addWidget(btn_save)

        layout.addLayout(btn_box)

    def _on_save(self) -> None:
        for i, edit in enumerate(self.name_inputs):
            name = edit.text().strip()
            if name:
                self.players[i].name = name
        self.accept()

    def get_configured_players(self) -> List[Player]:
        return self.players


# ============================================================================
# Main Window & HUD Coordinator
# ============================================================================

class TumblinDiceHUD(QMainWindow):
    """
    Tumblin' Dice Live Scoreboard Main Window.
    Surface-optimized live HUD featuring:
    - 4 Major Rounds snake turn manager
    - Global Spacebar 0 / Miss override
    - Live sidebar leaderboard
    - Turn Hero banner
    - Large tactile touch buttons
    - Board & Zone overlay
    """

    def __init__(
        self,
        state: Optional[GameState] = None,
        config_path: str = DEFAULT_CONFIG_PATH,
        camera_idx: Optional[int] = None,
        mock_feed: bool = False,
        quiescence_duration: float = 1.2,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Tumblin' Dice Automated Vision Scoreboard HUD")
        self.setMinimumSize(1024, 700)
        self.resize(1280, 800)
        self.setStyleSheet(SURFACE_QSS)

        self.state = state if state is not None else GameState()
        self.config_path = config_path
        self.camera_idx = camera_idx
        self.mock_feed = mock_feed
        self.quiescence_duration = float(quiescence_duration)
        self.vision_thread: Optional[VisionPipelineThread] = None

        # Mock feed automated timer
        self.mock_timer = QTimer(self)
        self.mock_timer.setInterval(2800)
        self.mock_timer.timeout.connect(self.trigger_mock_flick)
        self.mock_auto_running = False

        # Install global event filter to intercept Spacebar anywhere in the app
        app = QApplication.instance()
        if app:
            app.installEventFilter(self)

        self.init_ui()

        # Initialize background vision QThread if camera or mock feed specified
        if self.camera_idx is not None or self.mock_feed:
            self.init_vision_pipeline()

        self.refresh_all_views()

    def init_vision_pipeline(self) -> None:
        """Initializes and launches the background vision QThread."""
        self.vision_thread = VisionPipelineThread(
            camera_idx=self.camera_idx,
            mock_feed=self.mock_feed,
            config_path=self.config_path,
            quiescence_duration=self.quiescence_duration,
            parent=self,
        )
        self.vision_thread.frame_ready.connect(self.board_overlay.set_camera_frame)
        self.vision_thread.motion_status.connect(self.on_motion_status)
        self.vision_thread.settle_event.connect(self.on_vision_settle)
        self.vision_thread.calibration_status.connect(self.on_calibration_status)
        self.vision_thread.error_occurred.connect(self.on_vision_error)

        turn = self.state.get_current_turn()
        if turn:
            self.vision_thread.set_active_player(turn[0].id, turn[0].color)

        self.vision_thread.start()

    def init_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(12)

        # Top Section: Turn Hero Banner
        self.banner = TurnBannerWidget(self)
        root_layout.addWidget(self.banner)

        # Middle Section: Splitter with Leaderboard (Left) and Controls/Board (Right)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(8)

        # Left Column: Leaderboard
        self.leaderboard = LeaderboardWidget(self)
        splitter.addWidget(self.leaderboard)

        # Right Column: Board View + Touch Controls
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        # Board & Zone Preview (Expands to fill available vertical space)
        self.board_overlay = BoardOverlayWidget(config_path=self.config_path, parent=self)
        self.board_overlay.zone_selected.connect(self._on_board_zone_clicked)
        right_layout.addWidget(self.board_overlay, stretch=1)

        # Compact Touch Controls Panel (Fixed sleek height, no overlap or scroll)
        self.touch_controls = TouchScoreWidget(self)
        self.touch_controls.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.touch_controls.score_confirmed.connect(self.record_score)
        self.touch_controls.miss_triggered.connect(self.on_miss_override)
        self.touch_controls.undo_triggered.connect(self.on_undo_clicked)
        self.touch_controls.next_round_triggered.connect(self.advance_round)
        right_layout.addWidget(self.touch_controls, stretch=0)

        splitter.addWidget(right_container)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 5)

        root_layout.addWidget(splitter, stretch=1)

        # Bottom Bar: Status Bar & Fast Utilities
        bottom_bar = QHBoxLayout()
        bottom_bar.setSpacing(12)

        self.lbl_status = QLabel("Ready for flick. [Space] = 0/Miss  |  [Enter] = Confirm  |  [Ctrl+Z] = Undo")
        self.lbl_status.setStyleSheet("font-size: 14px; color: #94A3B8;")
        bottom_bar.addWidget(self.lbl_status, stretch=1)

        btn_setup = TouchButton("⚙ Players Setup", min_height=44, min_width=130)
        btn_setup.setStyleSheet("background-color: #334155; font-size: 14px;")
        btn_setup.clicked.connect(self.open_player_setup)
        bottom_bar.addWidget(btn_setup)

        btn_new_game = TouchButton("↻ New Match", min_height=44, min_width=120)
        btn_new_game.setStyleSheet("background-color: #334155; font-size: 14px;")
        btn_new_game.clicked.connect(self.confirm_new_game)
        bottom_bar.addWidget(btn_new_game)

        # Calibrator status badge
        self.lbl_cal_badge = QLabel("🎯 ArUco")
        self.lbl_cal_badge.setStyleSheet(
            "background-color: #1E293B; color: #94A3B8; font-weight: bold; font-size: 13px; "
            "padding: 4px 10px; border-radius: 6px; border: 1px solid #475569;"
        )
        bottom_bar.addWidget(self.lbl_cal_badge)

        if self.camera_idx is not None:
            self.lbl_cam_badge = QLabel(f"📷 CAM {self.camera_idx}")
            self.lbl_cam_badge.setStyleSheet(
                "background-color: #064E3B; color: #34D399; font-weight: bold; font-size: 13px; "
                "padding: 4px 10px; border-radius: 6px; border: 1px solid #059669;"
            )
            bottom_bar.addWidget(self.lbl_cam_badge)

            self.btn_force_settle = TouchButton("⚡ Settle Frame", min_height=44, min_width=120)
            self.btn_force_settle.setStyleSheet("background-color: #D97706; font-size: 14px; font-weight: bold;")
            self.btn_force_settle.clicked.connect(self.on_force_settle_clicked)
            bottom_bar.addWidget(self.btn_force_settle)

        if self.mock_feed:
            lbl_mock_badge = QLabel("⚡ MOCK FEED")
            lbl_mock_badge.setStyleSheet(
                "background-color: #1E3A8A; color: #60A5FA; font-weight: bold; font-size: 13px; "
                "padding: 4px 10px; border-radius: 6px; border: 1px solid #3B82F6;"
            )
            bottom_bar.addWidget(lbl_mock_badge)

            self.btn_mock_roll = TouchButton("🎲 Mock Roll", min_height=44, min_width=115)
            self.btn_mock_roll.setStyleSheet("background-color: #2563EB; font-size: 14px; font-weight: bold;")
            self.btn_mock_roll.clicked.connect(self.trigger_mock_flick)
            bottom_bar.addWidget(self.btn_mock_roll)

            self.btn_mock_toggle = TouchButton("▶ Auto Roll", min_height=44, min_width=115)
            self.btn_mock_toggle.setStyleSheet("background-color: #059669; font-size: 14px; font-weight: bold;")
            self.btn_mock_toggle.clicked.connect(self.toggle_mock_feed_auto)
            bottom_bar.addWidget(self.btn_mock_toggle)

        root_layout.addLayout(bottom_bar)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """
        Global Application Event Filter.
        Guarantees that Spacebar ALWAYS triggers the 0/Miss override from any focused widget,
        unless the user is actively typing inside a text input field (QLineEdit).
        """
        if event.type() == QEvent.Type.KeyPress:
            key_event: QKeyEvent = event  # type: ignore
            if key_event.key() == Qt.Key.Key_Space and not key_event.isAutoRepeat():
                if isinstance(watched, QLineEdit):
                    return False
                self.on_miss_override()
                return True
            elif key_event.key() == Qt.Key.Key_M and self.mock_feed and not key_event.isAutoRepeat():
                if not isinstance(watched, QLineEdit):
                    self.trigger_mock_flick()
                    return True
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Window-level keyboard shortcuts for rapid laptop or tablet keyboard use."""
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.on_miss_override()
            event.accept()
            return
        elif event.key() == Qt.Key.Key_M and self.mock_feed and not event.isAutoRepeat():
            self.trigger_mock_flick()
            event.accept()
            return
        elif event.key() in (Qt.Key.Key_1, Qt.Key.Key_2, Qt.Key.Key_3, Qt.Key.Key_4, Qt.Key.Key_5, Qt.Key.Key_6):
            pip_val = event.key() - Qt.Key.Key_0
            self.touch_controls.select_pip(pip_val)
            event.accept()
            return
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.touch_controls._on_confirm_clicked()
            event.accept()
            return
        elif (event.modifiers() & Qt.KeyboardModifier.ControlModifier) and event.key() == Qt.Key.Key_Z:
            self.on_undo_clicked()
            event.accept()
            return
        super().keyPressEvent(event)

    @pyqtSlot()
    def on_miss_override(self) -> None:
        """Triggered by Spacebar or touch button: records 0 points and advances to next player."""
        if self.state.is_game_over:
            return

        turn = self.state.get_current_turn()
        if not turn:
            return
        player, flick_num = turn

        record = self.state.record_miss()
        self.lbl_status.setText(f"❌ MISS (0 pts) recorded for {player.name} [Flick {flick_num}]")
        self._check_round_or_game_transition()
        self.refresh_all_views()

    @pyqtSlot(int, int, int)
    def record_score(
        self,
        score: int,
        pips: int,
        multiplier: int,
        die_x: Optional[float] = None,
        die_y: Optional[float] = None,
    ) -> None:
        """Records a scored flick and places a simulated die pin on the board visualizer."""
        if self.state.is_game_over:
            return

        turn = self.state.get_current_turn()
        if not turn:
            return
        player, flick_num = turn

        record = self.state.record_flick(score=score, pips=pips, multiplier=multiplier)

        # Place die pin on board
        if die_x is not None and die_y is not None:
            pin_x, pin_y = float(die_x), float(die_y)
        else:
            pin_x = float(200 + (flick_num * 80 + player.id * 35) % 400)
            pin_y = float(250 + (multiplier * 140) % 750)

        self.board_overlay.add_die_pin(pin_x, pin_y, player.color, pips, multiplier)

        # Sync placed die to mock vision thread if active
        if self.vision_thread is not None and self.mock_feed:
            self.vision_thread.add_mock_die(pips, multiplier, pin_x, pin_y, player.id)

        self.lbl_status.setText(
            f"✓ {player.name} scored {score} pts ({pips} pips × {multiplier}x) [Flick {flick_num}]"
        )
        self._check_round_or_game_transition()
        self.refresh_all_views()

    @pyqtSlot(object)
    def on_motion_status(self, motion_res: MotionResult) -> None:
        """Handles real-time motion state updates from background vision thread."""
        if self.state.is_game_over:
            return

        turn = self.state.get_current_turn()
        player_str = turn[0].name if turn else "Ready"

        if motion_res.state == MotionState.MOVING:
            self.lbl_status.setText(
                f"⚡ [ROLLING...] Active dice motion detected for {player_str} - Scoreboard frozen."
            )
            self.lbl_status.setStyleSheet("font-size: 14px; font-weight: bold; color: #F59E0B;")
        elif motion_res.state == MotionState.SETTLING:
            pct = int(motion_res.quiescence_progress * 100)
            self.lbl_status.setText(
                f"⏳ [SETTLING...] Waiting for dice stillness: {pct}% stillness ({motion_res.stillness_duration:.1f}s / {self.quiescence_duration:.1f}s)"
            )
            self.lbl_status.setStyleSheet("font-size: 14px; font-weight: 600; color: #38BDF8;")
        elif motion_res.state == MotionState.SETTLED:
            self.lbl_status.setStyleSheet("font-size: 14px; color: #34D399;")

    @pyqtSlot(object)
    def on_vision_settle(self, delta: DeltaScoringResult) -> None:
        """Handles real-time settle trigger from background vision thread with delta scoring."""
        if self.state.is_game_over or self.state.is_current_round_complete():
            return

        turn = self.state.get_current_turn()
        if not turn:
            return
        player, flick_num = turn

        if delta.is_miss:
            self.on_miss_override()
            self.lbl_status.setText(
                f"🎯 [VISION SETTLE] ❌ Gutter Miss (0 pts) recorded for {player.name} [Flick {flick_num}]"
            )
        else:
            die_x = delta.new_die.center[0] if delta.new_die else None
            die_y = delta.new_die.center[1] if delta.new_die else None
            self.record_score(
                score=delta.score,
                pips=delta.pips,
                multiplier=delta.multiplier,
                die_x=die_x,
                die_y=die_y,
            )
            self.lbl_status.setText(
                f"🎯 [VISION SETTLE] ✓ {player.name}: {delta.pips} pips × {delta.multiplier}x = {delta.score} pts [Flick {flick_num}]"
            )

    @pyqtSlot(bool)
    def on_calibration_status(self, is_calibrated: bool) -> None:
        """Updates ArUco homography lock badge."""
        if hasattr(self, "lbl_cal_badge"):
            if is_calibrated:
                self.lbl_cal_badge.setText("🎯 ArUco Warped")
                self.lbl_cal_badge.setStyleSheet(
                    "background-color: #064E3B; color: #34D399; font-weight: bold; font-size: 13px; "
                    "padding: 4px 10px; border-radius: 6px; border: 1px solid #059669;"
                )
            else:
                self.lbl_cal_badge.setText("🎯 ArUco Scan")
                self.lbl_cal_badge.setStyleSheet(
                    "background-color: #1E293B; color: #94A3B8; font-weight: bold; font-size: 13px; "
                    "padding: 4px 10px; border-radius: 6px; border: 1px solid #475569;"
                )

    @pyqtSlot(str)
    def on_vision_error(self, err_msg: str) -> None:
        """Handles background vision pipeline errors."""
        self.lbl_status.setText(f"[VISION ERROR] {err_msg}")
        self.lbl_status.setStyleSheet("font-size: 14px; font-weight: bold; color: #EF4444;")

    @pyqtSlot()
    def on_force_settle_clicked(self) -> None:
        """Referee manual override to force a vision settle pass."""
        if self.vision_thread is not None:
            self.vision_thread.force_settle()

    def _check_round_or_game_transition(self) -> None:
        """Checks if current major round ended or if game is complete."""
        if self.state.is_game_over:
            winners = self.state.get_winners()
            w_str = ", ".join(w.name for w in winners)
            self.lbl_status.setText(f"🏆 GAME OVER! Winner: {w_str} ({winners[0].total_score()} pts)")
            self.touch_controls.set_round_complete_mode(False)
        elif self.state.is_current_round_complete():
            next_r = self.state.current_round + 1
            self.lbl_status.setText(
                f"Round {self.state.current_round} complete! Clear physical dice from board and start Round {next_r}."
            )
            self.touch_controls.set_round_complete_mode(True, next_round_num=next_r)
            if self.mock_feed and hasattr(self, "mock_timer") and self.mock_timer.isActive():
                QTimer.singleShot(2200, self.advance_round)
        else:
            self.touch_controls.set_round_complete_mode(False)

    @pyqtSlot()
    def trigger_mock_flick(self) -> None:
        """Simulates an incoming computer vision detection event."""
        if self.state.is_game_over:
            if hasattr(self, "mock_timer") and self.mock_timer.isActive():
                self.mock_timer.stop()
            return

        turn = self.state.get_current_turn()
        if not turn:
            return
        player, flick_num = turn

        import random
        # 12% chance of miss / gutter
        is_miss = (random.random() < 0.12)
        if is_miss:
            self.on_miss_override()
            self.lbl_status.setText(f"[MOCK FEED] ❌ Gutter Miss (0 pts) recorded for {player.name} [Flick {flick_num}]")
        else:
            pips = random.randint(1, 6)
            mult = random.choice([1, 2, 3, 4])
            score = pips * mult
            self.record_score(score=score, pips=pips, multiplier=mult)
            self.lbl_status.setText(f"[MOCK FEED] 🎯 Detected {player.name}: {pips} pips × {mult}x = {score} pts [Flick {flick_num}]")

    @pyqtSlot()
    def toggle_mock_feed_auto(self) -> None:
        """Toggles automated roll feed timer on/off."""
        if not hasattr(self, "mock_timer"):
            return
        if self.mock_timer.isActive():
            self.mock_timer.stop()
            self.mock_auto_running = False
            if hasattr(self, "btn_mock_toggle"):
                self.btn_mock_toggle.setText("▶ Auto Roll")
                self.btn_mock_toggle.setStyleSheet("background-color: #059669; font-size: 14px; font-weight: bold;")
            self.lbl_status.setText("[MOCK FEED] Auto-roll paused. Click '🎲 Mock Roll' or press [M] to step.")
        else:
            self.mock_timer.start()
            self.mock_auto_running = True
            if hasattr(self, "btn_mock_toggle"):
                self.btn_mock_toggle.setText("⏸ Pause Feed")
                self.btn_mock_toggle.setStyleSheet("background-color: #D97706; font-size: 14px; font-weight: bold;")
            self.lbl_status.setText("[MOCK FEED] Auto-roll running (every 2.8s). Watch snake turns & round transitions!")

    @pyqtSlot()
    def advance_round(self) -> None:
        """Starts the next major round and clears board dice visualizer."""
        if self.state.advance_to_next_round():
            self.board_overlay.clear_dice()
            if self.vision_thread is not None:
                self.vision_thread.reset_round()
            self.lbl_status.setText(
                f"Round {self.state.current_round} began! Order: {self.state.get_round_direction_name(self.state.current_round)}"
            )
            self._check_round_or_game_transition()
            self.refresh_all_views()

    @pyqtSlot()
    def on_undo_clicked(self) -> None:
        """Rolls back the most recent flick score."""
        if self.state.undo():
            if self.vision_thread is not None:
                self.vision_thread.undo_flick()
            self.lbl_status.setText("Undid last flick action.")
            self._check_round_or_game_transition()
            self.refresh_all_views()
        else:
            self.lbl_status.setText("Nothing to undo.")

    @pyqtSlot(int)
    def _on_board_zone_clicked(self, multiplier: int) -> None:
        """Tapping on a board zone selects that multiplier in touch controls."""
        self.touch_controls.select_multiplier(multiplier)
        self.lbl_status.setText(f"Selected {multiplier}x zone from board view.")

    def open_player_setup(self) -> None:
        """Opens touch player configuration dialog."""
        dialog = PlayerSetupDialog(self.state.players, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_players = dialog.get_configured_players()
            self.state.reset_game(new_players)
            self.board_overlay.clear_dice()
            if self.vision_thread is not None:
                self.vision_thread.reset_round()
            self.refresh_all_views()
            self.lbl_status.setText("Player roster updated. New game started.")

    def confirm_new_game(self) -> None:
        """Prompts to restart the match."""
        reply = QMessageBox.question(
            self,
            "Reset Match",
            "Reset scores and start a fresh 4-round match?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.state.reset_game()
            self.board_overlay.clear_dice()
            if self.vision_thread is not None:
                self.vision_thread.reset_round()
            self.refresh_all_views()
            self.lbl_status.setText("Game reset. Round 1 begins!")

    def refresh_all_views(self) -> None:
        """Synchronizes all UI components with the current GameState."""
        self.banner.update_view(self.state)
        self.leaderboard.update_view(self.state)
        if self.vision_thread is not None:
            turn = self.state.get_current_turn()
            if turn:
                self.vision_thread.set_active_player(turn[0].id, turn[0].color)

    def close(self) -> bool:
        """Ensures background vision worker thread is stopped even if window wasn't visible."""
        if hasattr(self, "mock_timer") and self.mock_timer.isActive():
            self.mock_timer.stop()
        if hasattr(self, "vision_thread") and self.vision_thread is not None:
            self.vision_thread.stop()
            self.vision_thread = None
        return super().close()

    def closeEvent(self, event: Any) -> None:
        """Ensures background vision worker thread is stopped cleanly."""
        if hasattr(self, "mock_timer") and self.mock_timer.isActive():
            self.mock_timer.stop()
        if hasattr(self, "vision_thread") and self.vision_thread is not None:
            self.vision_thread.stop()
            self.vision_thread = None
        super().closeEvent(event)


# ============================================================================
# Main Entry Point
# ============================================================================

def main() -> None:
    """Application launcher."""
    import argparse
    parser = argparse.ArgumentParser(description="Tumblin' Dice Live Scoreboard HUD with Automated Computer Vision")
    parser.add_argument("--players", type=int, default=4, help="Number of players (1..10)")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to zones_config.json")
    parser.add_argument("--fullscreen", action="store_true", help="Launch in fullscreen mode")
    parser.add_argument(
        "--camera",
        type=int,
        nargs="?",
        const=0,
        default=None,
        help="Camera device index (e.g. 0, 1) for live overhead video feed",
    )
    parser.add_argument(
        "--mock-feed",
        action="store_true",
        help="Launch HUD with simulated roll feed and interactive mock controls",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Alias for --mock-feed interactive software simulation mode",
    )
    parser.add_argument(
        "--quiescence",
        type=float,
        default=1.2,
        help="Motion quiescence stillness settle duration in seconds (default: 1.2s)",
    )
    args = parser.parse_args()

    mock_feed = bool(args.mock_feed or args.demo)

    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    state = GameState()
    state._init_default_players(args.players)

    window = TumblinDiceHUD(
        state=state,
        config_path=args.config,
        camera_idx=args.camera,
        mock_feed=mock_feed,
        quiescence_duration=args.quiescence,
    )
    if args.fullscreen:
        window.showFullScreen()
    else:
        window.show()

    if mock_feed:
        # Start auto roll automatically when launched with --mock-feed
        window.toggle_mock_feed_auto()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
