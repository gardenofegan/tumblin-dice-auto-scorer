# Tumblin' Dice — Computer Vision Scoring HUD

An automated computer vision scoreboard and spectator HUD for the classic tabletop game **Tumblin' Dice**, optimized for Microsoft Surface Pro 9 and overhead camera setups.

---

## 🧰 Hardware Requirements

1. **Overhead USB Camera**:
   - Standard USB webcam or document camera mounted directly above the Tumblin' Dice board facing straight down.
   - 720p or 1080p resolution recommended.
2. **Display / Host Device**:
   - Microsoft Surface Pro or Windows PC/laptop running Windows 10/11.
   - Touchscreen capability enables direct score tapping and color sampling.
3. **Optional Projector / Spectator Screen**:
   - Secondary monitor or projector connected via HDMI/USB-C for crowd scoreboards.
4. **Tumblin' Dice Board & Dice Set**:
   - Standard 4-tier wooden stepped board (0x launch deck, 1x, 2x, 3x, 4x stop troughs).
   - Up to 10 distinct colored sets of standard 6-sided dice (D6).

---

## 🚀 Quickstart & Launch Options

### Live Overhead Camera Mode
```powershell
python app.py --camera 0
```
*Note: If your Surface Pro has a built-in front/rear camera, the external USB camera might be device index `1`. Try `python app.py --camera 1`.*

### Fullscreen Operator / Kiosk Mode
```powershell
python app.py --camera 0 --fullscreen
```

### Mock / Simulation Demo (No Camera Required)
```powershell
python app.py --mock-feed
```
Launches an interactive software simulation with synthetic dice rolls, animated settle cycles, and full operator controls.

### Command Line Options
| Option | Description |
| :--- | :--- |
| `--camera [N]` | Camera device index (`0`, `1`, etc.) for live video capture |
| `--mock-feed` / `--demo` | Simulated video feed with synthetic dice for testing without hardware |
| `--fullscreen` | Launch in fullscreen kiosk mode |
| `--players N` | Initial player count (1 to 10, default: 4) |
| `--quiescence S` | Motion stillness settle duration in seconds (default: `1.2s`) |
| `--config PATH` | Custom path to board zones configuration JSON |

---

## 📐 Board Calibration & Perspective Alignment

When setting up your camera above the wooden board for the first time:

1. **Corner Calibration**:
   - In the HUD, open **Calibration** mode (or tap the gear icon).
   - Click the 4 corners of the wooden board in clockwise order:
     - Top-Left (Launch tray edge)
     - Top-Right (Launch tray edge)
     - Bottom-Right (4x trough edge)
     - Bottom-Left (4x trough edge)
   - The computer vision engine warps the camera view into a normalized 2:3 vertical aspect ratio.
2. **Zone Division**:
   - The HUD projects the scoring zones:
     - **0x Launch Deck**: Top tier where players rest and flick their dice.
     - **1x, 2x, 3x Levels**: Stepped descending platforms.
     - **4x Stop Trench**: Bottom gutter for maximum multiplier.
3. **0x Launch Zone Gating**:
   - Dice placed on the 0x launch deck are **completely suppressed** by the computer vision pipeline. Players can rest dice, adjust fingers, and prepare shots without triggering false roll detections or turn advances.

---

## 🎨 Player Setup & Interactive Video Color Dropper

1. Tap **Player Setup** on the left column.
2. Set the player count (1 to 10 players) using the **[+]** and **[-]** buttons.
3. Assign dice colors:
   - Tap the **Color Dropper** icon next to any player.
   - Place their die on the board under the camera (or click directly on the die in the live video feed).
   - The system instantly samples the average HSV hue/saturation and binds that die profile to the player.

---

## 🎯 Gameplay & Scoring Flow

1. **Rolling**:
   - The active player flicks their die from the 0x launch deck down the stepped board.
2. **Motion Settle & Auto-Scoring**:
   - The CV engine tracks optical flow and waits for **quiescence** (1.2 seconds of zero motion).
   - Once settled, the engine detects:
     - The die's **scoring zone multiplier** (1x, 2x, 3x, 4x).
     - The die's **top pip value** (1–6).
     - Multiplies `pips × multiplier` and attributes points to the player.
3. **Touch Keypad Manual Override**:
   - The right column provides a giant **Touch Keypad**:
     - **0 / Miss**: Immediately marks a roll as off the board or scratch.
     - **Numeric keys**: Allows the operator or player to adjust or confirm any roll with a single tap.
     - **Confirm**: Advances turn to the next player.
