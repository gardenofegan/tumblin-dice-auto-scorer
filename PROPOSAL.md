# 🎲 Proposal: Tumblin' Dice Automated Vision Scoreboard

**Task ID:** `TASK-102`  
**Status:** `APPROVED` (Promoted to active execution queue)  
**Stage:** `BUILD`  
**Project Path:** `C:\Users\garde\Documents\lab\tumblin_dice`  

---

## 💡 Concept & Goal

Build a desktop application (PyQt6 + OpenCV + YOLOv8/HSV color sampling) that uses an overhead USB webcam to automatically score games of **Tumblin' Dice** in real time.

### The Game & Scoring Rules
- **Players:** Up to 10 players, each assigned 4 dice of a distinct color (up to 40 dice total).
- **Structure:** 4 major rounds. Each major round consists of 4 turns per player (one flick per turn).
- **Turn Order:** Snake draft reversing each major round:
  - Round 1: Player 1 ➔ Player N
  - Round 2: Player N ➔ Player 1
  - Round 3: Player 1 ➔ Player N
  - Round 4: Player N ➔ Player 1
- **Score Calculation:** `Dice Pip Face (1–6) × Zone Multiplier (e.g., 1×, 2×, 3×, 4×)`
- **Dynamic Physics:** Flicked dice can knock opponent dice into higher/lower zones or off the board. Off-board dice score 0.
- **Winner:** Highest cumulative score after 4 major rounds.

---

## 🏗️ Proposed Architecture & Vision Pipeline

```
              Overhead USB Webcam (1080p @ 30 FPS)
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │ 1. Motion Gate (Frame Differencing)          │
        │    - Moving dice detected? ➔ Show "Rolling"  │
        │    - Motion settled > 1.2s? ➔ Trigger Score  │
        └──────────────────────┬───────────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │ 2. ArUco Auto-Homography                     │
        │    - 4 corner fiducials (DICT_4X4_50)        │
        │    - Planar warp eliminates camera drift     │
        │    - Pre-calibrated scoring zones snap on    │
        └──────────────────────┬───────────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │ 3. Dice & Pip Detection                      │
        │    - Locate dice bounding boxes & top pips   │
        │    - HSV color clustering matches player     │
        │    - Point-in-polygon assigns multiplier     │
        └──────────────────────┬───────────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │ 4. PyQt6 Live HUD & Leaderboard              │
        │    - Active player highlight & turn badge    │
        │    - Big Spacebar / UI button for "0 / Miss" │
        │    - Live colored zone overlays on feed      │
        └──────────────────────────────────────────────┘
```

---

## 👤 Human Prerequisites & Hardware Setup

Before automated code runs, you'll need:
- [ ] Overhead USB webcam mounted above board center (rigid mount/clamp recommended).
- [ ] 4 ArUco fiducial markers printed & taped to the 4 corners of the board floor (`DICT_4X4_50`).
- [ ] Colored dice sets (4 dice per player, distinct colors).
- [ ] Diffused overhead lighting to minimize harsh shadows.

---

## 📋 Proposed Step-by-Step Task Breakdown

When approved, this project will be broken into sequential, testable `[TODO]` milestones:

1. **TASK-102A (Calibration & ArUco Warp):**
   - OpenCV script to detect 4 ArUco markers, compute homography matrix, and warp perspective into a normalized 2D plane.
   - Mouse-click polygon zone editor (`zones_config.json`) with persistent save/load.
2. **TASK-102B (Motion Gating & State Machine):**
   - Frame-differencing motion detector that waits for physical dice to settle (>1.2s quiescence) before triggering CV scoring.
3. **TASK-102C (Color Sampling & Dice Detection):**
   - HSV color classifier to match dice bounding boxes to configured player dice profiles (`player_profiles.json`).
   - Pip counter filter (1–6) on the top face.
4. **TASK-102D (PyQt6 Game HUD & Score Engine):**
   - 4-round snake order manager (1➔N, N➔1), leaderboard table, and quick Spacebar "0 / Off-Board" override.

---

## ❓ Alignment Questions for You

Before promoting to `[TODO]` and writing code:
1. **Camera Feed:** Do you want the scoreboard UI and webcam feed combined in a single desktop window (e.g. PyQt6), or a lightweight local web dashboard (FastAPI/Flask + browser)?
2. **Dice Detection Strategy:** Would you prefer starting with a lightweight classical OpenCV approach (contour + HSV color masking + pip blob detection) for zero-latency local setup, or jump straight to training a YOLOv8 model on your specific dice?
3. **Board Zones:** How many multiplier zones does your custom board have (e.g., 1×, 2×, 3×, 4×)?

---
*To approve this plan and promote to execution:*
`python inbox_processor.py --approve TASK-102` (or simply say *"Approve TASK-102"* in chat)
