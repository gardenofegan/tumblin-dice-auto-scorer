Automating Tumblin' Dice Scoreboard

https://gemini.google.com/app/66926944dea89073

User prompt: Using a webcam, I want to be able to make a live scoreboard to help me to score automatically games of Tumblin Dice. I have a semi custom board. The point of the game is each player has 4 dice of a single color. 1 at a time players take turn flicking these onto the board. Each area of the board is worth a different amount and the amount is the area of the board times the dice face showing. The scoreboard should stay up to date after each die roll / flick. How can I best accomplish this?

Response: Automating the scoring for Tumblin' Dice is a classic "Computer Vision" (CV) challenge. Since the game relies on two distinct variables—the physical location of the die and the pip value on the top face—you need a system that can handle both spatial mapping and object classification.

Here is a breakdown of the most effective way to build this.

1. The Hardware Setup

For CV to work reliably, consistency is your best friend.

Top-Down Camera (Bird's Eye View): Mount your webcam directly over the board. A "flat" perspective removes the need for complex 3D math and makes it easier to define the "scoring zones."

Consistent Lighting: Shadows can be mistaken for pips. Try to use diffused lighting (like a ring light or a desk lamp with a paper shade) to minimize harsh glares on the dice.

High Contrast: Since you have a semi-custom board, ensure the scoring area boundaries are clear. If the board and dice are similar colors, detection becomes much harder.

2. The Vision Pipeline

The best approach today is to use Python with OpenCV for image processing and a machine learning model (like YOLOv8) for the dice.

Step A: Define the Scoring Zones

Instead of trying to "teach" the computer what the board looks like, use Perspective Transformation. You will click the four corners of your board in the camera feed, and the software will "warp" that image into a perfect 2D rectangle.

Once warped, you can hard-code the coordinates for each multiplier zone (1x,2x,3x,4x).

Step B: Detect the Dice (YOLOv8)

Standard OpenCV "blob detection" often fails with dice because the pips confuse the shape. A pre-trained YOLO (You Only Look Once) model is better.

Detection: The model identifies the bounding box of each die.

Color Classification: Since each player has a specific color, you can look at the average pixel color within that bounding box to assign the die to a player.

Pip Counting: A second specialized model (or a simple contour filter) counts the pips on the top face of the die.

3. Logic & Scoreboard

The "Live" aspect requires a state machine to prevent the score from jumping around while a die is still rolling.

Component

Logic

Motion Detection

The system should only "calculate" when the board has been still for ∼1 second.

Zone Mapping

Take the (x,y) center of a detected die and check which polygon (zone) it sits in.

Calculation

(Face Value) * (Zone Multiplier) = Points.

The Display

Use Pygame or a simple Flask web app to display a "Live Leaderboard" on a monitor or tablet next to the board.

4. Recommended Tools

Language: Python 3.x

Libraries: * cv2 (OpenCV) for camera handling and warping.

ultralytics (YOLOv8) for high-speed dice and pip detection.

numpy for the math behind the scoring zones.

Dataset: You can find "Dice" datasets on Roboflow to train your model so it recognizes your specific dice colors and styles.

A Simplified "No-Code" Starting Point

If you aren't ready to dive into Python, you could look at Anyline or Google Lens-style OCR, but for a custom board and live movement, the Python/OpenCV route is the only one that will provide the "automated" feel you're looking for.

Would you like me to draft a basic Python script to help you start detecting the board corners and warping the view?

User prompt: This sounds like a good start. Is there a way for me to define the corners of the board each time I start the app / game? As you mention in Step A

Response: Yes, absolutely. In fact, defining the corners manually at the start of each session is the gold standard for DIY sports or board game "VAR" (Video Assistant Referee) systems. It accounts for the webcam being bumped or the table being shifted slightly.

The most user-friendly way to do this in Python is using OpenCV’s mouse callback function.

The "Click-to-Calibrate" Workflow

You can build a small "Setup Mode" that runs before the game logic starts. Here is how that looks conceptually:

Freeze a Frame: The app captures one still image from the webcam.

User Input: You click the four outer corners of your Tumblin' Dice board in a specific order (e.g., Top-Left, Top-Right, Bottom-Right, Bottom-Left).

Calculate Matrix: The code uses cv2.getPerspectiveTransform to calculate how to "stretch" that skewed shape into a perfect flat rectangle.

Warp: Every frame from the live video is then passed through that transformation.

A Simple Starter Script

Here is a "boilerplate" Python snippet to handle those clicks and perform the warp. You’ll need opencv-python and numpy installed.

import cv2import numpy as np# List to store our 4 corner pointspoints = []def click_event(event, x, y, flags, params):    if event == cv2.EVENT_LBUTTONDOWN:        points.append((x, y))        # Draw a small circle where you clicked        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)        cv2.imshow("Calibration", img)# 1. Capture a frame for calibrationcap = cv2.VideoCapture(0)ret, img = cap.read()cv2.imshow("Calibration", img)cv2.setMouseCallback("Calibration", click_event)print("Click the 4 corners of the board: TL, TR, BR, BL. Press 'q' when done.")while len(points) < 4:    cv2.waitKey(1)# 2. Define the "Perfect" board dimensions (e.g., 500x800 pixels)width, height = 500, 800pts1 = np.float32(points)pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])# 3. Create the Transformation Matrixmatrix = cv2.getPerspectiveTransform(pts1, pts2)# 4. Apply to Live Feedwhile True:    ret, frame = cap.read()    # This is the "Magic" line that flattens your board    warped_board = cv2.warpPerspective(frame, matrix, (width, height))        cv2.imshow("Live Scoreboard View", warped_board)    if cv2.waitKey(1) & 0xFF == ord('q'):        breakcap.release()cv2.destroyAllWindows()

Why this is better for Tumblin' Dice

Since Tumblin' Dice boards are tiered (stepped), a standard "flat" warp might have a little bit of "parallax error" (where the dice on the lower tiers look slightly shifted).

Pro-Tip for your semi-custom board: Instead of just clicking the 4 outside corners, you could actually define each scoring tier as its own separate rectangle.

Tier 1 (1x): Click 4 corners.

Tier 2 (2x): Click 4 corners.

This makes the "Is the die inside the zone?" math much more accurate because you aren't guessing where the steps are based on a 2D projection.

Would you like me to show you how to draw those "Scoring Zones" on the screen so you can confirm the multipliers are in the right spot?

User prompt: The scoring zones sounds correct as I have 6 unique scoring zones. Also, they all aren't exactly rectangle, is there a way to select 6 or even 8 points to help draw not a perfect rectangle for the zone?

Response: Since your Tumblin' Dice board has 6 unique and non-rectangular zones, you've hit the most common hurdle in board game CV: Irregular Polygons.

Instead of a simple "Point-in-Rectangle" check, you can use OpenCV’s pointPolygonTest. This allows you to define a zone with as many points as you want (4, 6, 8, or more) and the computer will mathematically tell you if a die's center is inside that "fence."

Here is how to upgrade your logic to handle those custom zones.

1. Defining "Multi-Point" Zones

Rather than just 4 corners for the whole board, you will create a "Zone Definition" phase. For each of your 6 zones, you'll click the points that outline it.

The Logic:

Select Zone: Tell the app "I am defining Zone 1 (1×)."

Click Vertices: Click every corner of that specific area.

Close Polygon: Store those points in a list.

Repeat: Do this for all 6 zones.

2. The Python Implementation

This snippet uses the cv2.pointPolygonTest function. This is the "magic" function that handles irregular shapes.

import cv2import numpy as np# A list of dictionaries to store each zone's name and its pointszones = [    {"name": "Zone 1 (1x)", "points": [], "multiplier": 1},    {"name": "Zone 2 (2x)", "points": [], "multiplier": 2},    # ... add more zones as needed]current_zone_idx = 0def click_event(event, x, y, flags, params):    global current_zone_idx    if event == cv2.EVENT_LBUTTONDOWN:        zones[current_zone_idx]["points"].append([x, y])        # Draw the point so you can see your progress        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)        cv2.imshow("Calibration", img)# --- CALIBRATION PHASE ---# (Assume 'img' is a captured frame from your webcam)# For each zone, you click the vertices and press 'n' for next zonewhile current_zone_idx < len(zones):    print(f"Define {zones[current_zone_idx]['name']}. Press 'n' for next.")    # (Mouse callback logic here...)    # Once points are collected and 'n' is pressed:    # Convert points to a numpy array for OpenCV    # zones[current_zone_idx]["poly"] = np.array(zones[current_zone_idx]["points"], np.int32)# --- LIVE SCORING PHASE ---# Inside your live loop, once you detect a die at (die_x, die_y):def find_zone(die_x, die_y):    for zone in zones:        # result > 0 means the point is inside the polygon        is_inside = cv2.pointPolygonTest(zone["poly"], (die_x, die_y), False)        if is_inside >= 0:            return zone["multiplier"]    return 0 # Off the board

3. Handling the "Tiered" (3D) Problem

Because Tumblin' Dice is a tiered/stepped game, a die on the "bottom" tier is physically further from the camera than a die on the "top" tier. This causes Parallax Error: the die might look like it's in the 2× zone when it's actually in the 1× zone because of the angle.

To fix this on your custom board:

Keep the camera high: The higher the webcam, the less the "steps" will distort the position.

Define zones based on the floor: When clicking your zones, always click the "floor" of that tier, not the top of the railings. This ensures the base of the die is what triggers the score.

4. Prototyping Tip: The "Visual Confirmation"

Since your zones are irregular, it helps to have the app draw a semi-transparent colored overlay on the live feed.

1× zone = Blue tint

2× zone = Green tint

3× zone = Red tint

This way, if the score looks wrong, you can instantly see if a zone boundary was drawn slightly off.

Would you like me to show you how to draw those colored "overlays" so you can visually verify your custom zones?

User prompt: Yes let's definitely have colored overlays to make sure all is going well.

Response: Adding colored overlays is a great move. It turns your software from a "black box" into a clear, visual map. This is especially helpful in Tumblin' Dice because you can instantly see if the "drop-off" between tiers is being handled correctly by your camera's perspective.

In OpenCV, we do this by creating a mask—basically a transparent drawing layer—and then blending it with your live video feed using cv2.addWeighted.

1. The "Transparent Overlay" Logic

To make this work with your irregular zones, you'll use cv2.fillPoly to color in the shapes you clicked.

Here is how you can implement the overlay rendering:

import cv2import numpy as npdef draw_zones(frame, zones):    # Create a blank overlay the same size as the frame    overlay = frame.copy()        # Define some colors for the multipliers (B, G, R format)    colors = [        (255, 0, 0),    # Blue (1x)        (0, 255, 0),    # Green (2x)        (0, 0, 255),    # Red (3x)        (255, 255, 0),  # Cyan (4x)        (255, 0, 255),  # Magenta (Bonus)        (0, 255, 255)   # Yellow (Penalty)    ]    for i, zone in enumerate(zones):        if len(zone["points"]) > 2:            pts = np.array(zone["points"], np.int32)            # Fill the polygon on the overlay layer            cv2.fillPoly(overlay, [pts], colors[i % len(colors)])    # Blend the overlay with the original frame    # alpha = 0.7 (original frame weight), beta = 0.3 (overlay transparency)    alpha = 0.7    output = cv2.addWeighted(frame, alpha, overlay, 1 - alpha, 0)    return output

2. How it works in your "Live" loop

Your main program loop will now follow this sequence:

Grab the raw frame from the webcam.

Detect the dice (getting their x,y coordinates).

Draw the colored zone overlays onto the frame.

Draw a circle or a "score bubble" over each detected die so you know which zone it's triggering.

Display the final "HUD" (Heads-Up Display) to you.

3. Visual Debugging Tips

Since your board is semi-custom, look for these two things on your screen once the overlays are live:

The "Edge" Test: If a die is sitting right on the line between the 2× and 3× zones, does the colored overlay match where the die physically is? If not, you may need to re-click your points.

Perspective Shift: Because of the webcam angle, a die on a higher tier might "lean" over the line of a lower tier. Always draw your zone boundaries based on the "floor" of the board, not the top of the walls.

4. Extra Polish: The "Active" Glow

If you want to get fancy, you can make the zone "light up" (increase the transparency or change color) only when a die is detected inside it.

# Inside the loop for each zone:is_occupied = any(cv2.pointPolygonTest(zone_poly, (d_x, d_y), False) >= 0 for d_x, d_y in detected_dice)if is_occupied:    # Use a brighter version of the color or higher alpha    cv2.fillPoly(overlay, [pts], active_color)

What would you like to tackle next? We can look at the "Dice Detection" side (counting the pips) or how to build the "Leaderboard" UI that shows the players' totals.

User prompt: Great. Take all the details we have above. Document out a requirements document that I can pass to a coding assistant to help me build this out. It needs to have all the live scoring functionality, the zone setting functionality and feedback (overlays), as well as anything else required to make this application work smoothly. For the way we play the game there are 4 major rounds pf scoring. Each round players take turns flicking a dice onto the playing field. So there are 4 rounds within each major round. The scoring for each round is "set" once everyone has flicked their dice for that major round. The player directions changes each major round (so for example if there are 4 players, round one is 1 -> 4, round 2 is 4 ->1, round 3 is 1->4 and round 4 is 4-> 1. ) After the 4 major rounds whomever has the highest score wins. At the beginning of each game help me setup the camera and the scoring zones, help me setup the players. their names and their dice color. Generally I can support up to 10 players so that means there could be up to 10 different colored dice on the board and up to 40 unique dice (4 of each color) to score. If a player flicks their dice and it goes off the board that is a 0. Hopefully this can be tracked but if there is a miss, maybe there needs to be a "next player" button as well in case the camera misses that player got a 0 because their flick was off the board enough to not be captured by the camera. Flicking mid round can also cause other players dice to move or go off the board so the board state is very dynamic throughout the major rounds. The app should be super user friendly, should always be showing the score for all players as well as clearly denote who's turn it currently is. Help me produce a step by step requirements document I can share with the coding assistant to get this app up and running.

Response: This is a fantastic project. Combining Computer Vision with a physical tabletop game like Tumblin' Dice requires a mix of spatial logic and state management.

Below is a comprehensive Technical Requirements Document (TRD). You can hand this directly to a coding assistant (like me in a new session or another developer tool) to begin the implementation.

Requirements Document: Tumblin' Dice Automated Scorer

1. Project Overview

The goal is a Python-based desktop application that uses a webcam to automatically track and score "Tumblin' Dice." The app must identify dice values, locations (zones), and player colors to maintain a live leaderboard across a multi-round game.

2. Hardware & Environment

Camera: Top-down (Bird's Eye) webcam mount.

Lighting: Diffused, consistent lighting to minimize shadows.

Game Pieces: Up to 10 players, each with 4 dice of a unique color.

Board: Semi-custom, tiered board with 6 unique, irregular scoring zones.

3. Phase 1: Setup & Calibration (The "Pre-Game" Mode)

The app must start in a setup mode with the following steps:

A. Camera Calibration

Live View: Show the webcam feed.

Zone Definition: The user must be able to click points to define 6 unique polygons (scoring zones).

Multiplier Assignment: For each polygon, the user assigns a multiplier (e.g., 1×,2×,3×,4×,0×).

Visual Feedback: Draw semi-transparent colored overlays for each defined zone.

B. Player Registration

Support: Up to 10 players.

Inputs: Name and Dice Color.

Color Sampling: The user should be able to hold a die under the camera and "sample" the color to ensure the CV model knows exactly what "Red" or "Blue" looks like under current lighting.

4. Phase 2: Game Logic & State Machine

The app must manage the specific turn-based structure of Tumblin' Dice.

A. Tournament Structure

4 Major Rounds: * Round 1 & 3: Forward order (Player 1 → Player 10).

Round 2 & 4: Reverse order (Player 10 → Player 1).

Turns: Within each Major Round, every player flicks 4 dice (1 at a time).

Finality: A player's score for the round is only "locked" once all players have finished their flicks for that Major Round.

B. The "Flick" Lifecycle

Motion Detection: The system waits for the board to be "Still."

Detection: Identify all dice on the board.

Calculation: (Top Face Value) * (Zone Multiplier).

Tracking: If a die previously on the board is knocked off by a new flick, the score must update automatically.

5. Phase 3: Computer Vision Pipeline

Model: Use YOLOv8 (or similar) to detect dice bounding boxes and pip counts.

Color Classification: Use HSV Color Masking or a K-Nearest Neighbors (KNN) classifier to assign each detected die to a player based on the sampled registration colors.

Spatial Mapping: Use cv2.pointPolygonTest to determine which zone a die’s center (x,y) coordinate occupies.

Out of Bounds: Any die not within a defined polygon is treated as 0.

6. Phase 4: User Interface (The "Live HUD")

Dashboard: Always visible sidebar showing the total scoreboard.

Turn Indicator: Clearly highlight whose turn it is and how many dice they have left to flick (e.g., "Player 2: Flick 3 of 4").

Manual Overrides: * "Next Player" Button: To skip or advance if a die is flicked entirely off-camera.

"Manual Edit": A way to click a die on screen and manually correct the value if the CV miscounts pips.

Live Overlays: Colored zone overlays should remain visible (at low opacity) during gameplay to confirm die placement.

7. Technical Stack Requirements

Language: Python 3.10+

Core Libraries: opencv-python, numpy, ultralytics (YOLO), custominter or PyQt6 (for the GUI).

Persistence: Save/Load calibration settings so the user doesn't have to redraw zones if the camera hasn't moved.

Suggested Next Step

Would you like me to generate the YOLO training guide (how to get the dice recognition working) or the State Machine code that handles the "1 to 4, 4 to 1" turn logic?

User prompt: Besides handing this document to a coding assistant, what all steps do I need to take? Do I need to do something special for YOLO training?

Response: Beyond just handing over the document, your success depends on the quality of the "eyes" you give the AI. Computer vision is only as good as the data it's trained on.

Here is the "Missing Manual" for the steps you'll need to take personally to get this running smoothly.

1. The Environment Setup

Before your coding assistant writes a single line, you need a playground for the code to live in.

Install Python 3.10+: Download it from python.org.

VS Code: This is the most popular editor. Install the "Python" extension.

Install Core Libraries: Open your terminal and run: pip install ultralytics opencv-python numpy custominter

The Model: You’ll likely start with yolov8n.pt (the "Nano" version). It's incredibly fast—crucial for "live" scoring—but you’ll need to train it on your specific dice.

2. The "Special" Part: YOLO Training

You asked if there’s something special for YOLO training. Yes. Detecting a "cube" is easy, but reading "pips" (the dots) accurately from a distance is the hard part.

The Data Collection Secret

Don't just download "dice photos" from the internet. Your model needs to see your board, under your lights, with your dice.

Record Video: Use your mounted webcam to record 5 minutes of you flicking dice. Move them around, knock them into each other, and place them in every scoring zone.

Extract Frames: Take about 200–300 still images from that video. This ensures the background (your board) is a constant the AI learns to ignore.

Labeling Strategy (The "Special" Sauce)

When you use a tool like Roboflow (highly recommended for beginners) to label your images, you have two choices:

Option A (Single Model): You draw a box around the whole die and label it "6", "5", etc. This is easier but requires more images (500+) to be accurate.

Option B (Two-Stage - Recommended): 1. Train one model to just find "Die". 2. Use a second "Pip Counter" logic that zooms in on that box and counts the dots. Actually, for Tumblin' Dice, the easiest way is to label 6 classes: side1, side2, side3, side4, side5, side6.

3. Physical Calibration Steps

Even with the best code, a shaky camera will ruin the score.

Rigid Mount: If your webcam is on a flexible arm, the "Zones" you drew will drift every time someone bumps the table. Use a C-Stand or a bolt-on desk mount.

The "Ground" Principle: When you draw your scoring zones in the app, draw them on the floor of the board. If you draw them at the height of the walls, a die sitting on the bottom tier might look like it's in the tier above it because of the camera's angle.

4. Testing the "Game State"

Since you have 4 major rounds and 10 players, the "logic" can get messy.

Dry Run: Before you trust the CV, have the coding assistant build a "Manual Mode" where you can click buttons to simulate turns.

The "Order" Test: Verify that the app correctly flips from Player 1-10 to 10-1 for Round 2.

Summary Checklist

Step

Task

Why it matters

1

Mount Camera

Perspective must be identical for calibration and play.

2

Record "Flick" Video

Provides the raw frames for the AI to learn from.

3

Label in Roboflow

Teaches the AI what "1" through "6" looks like on your dice.

4

Run Training

Use a free tool like Google Colab to train the model (it's faster than your laptop).

5

Integrate & Test

Plug the .pt file into the Python app and start flicking.

Would you like me to walk you through how to use Google Colab to train your model for free once you have your photos ready?

Sources

Exploring data labeling for computer vision projects - Ultralytics

User prompt: Give me a detailed walk through of all the above, step by step

Response: This project is the perfect blend of physical gaming and tech. To make this work, you’ll be moving through three distinct phases: Physical Setup, AI Training (The "Brain"), and Application Assembly.

Here is your master step-by-step roadmap to building the Tumblin' Dice Automated Scorer.

Phase 1: The Physical Foundation

The AI can only be as accurate as the "eyes" you give it.

Mount the Camera: You need a "Bird's Eye" view. Secure your webcam directly over the center of the board.

Pro Tip: Use a rigid mount (like a desk clamp or C-stand). If the camera moves by even a centimeter after you define your zones, the scoring will be off.

Diffuse the Lighting: Dice are shiny. Harsh lights create "specular highlights" (white glares) that look like pips to an AI. Use a ring light or place a piece of parchment paper over your desk lamp to soften the light.

Color Sampling: Gather one die of each color you plan to support (up to 10). You will need these to "teach" the app which player is which.

Phase 2: Teaching the AI (YOLO Training)

You need to create a custom "Weights" file (.pt) that tells the computer: "This is a die, and it's showing a 4."

Capture Dataset Video: Record 2–3 minutes of "gameplay." Move the dice around, stack them, and flick them. This gives you images of the dice from every possible angle on your specific board.

Extract Frames: Use a tool like Roboflow (free for public projects) to upload your video and extract roughly 300 still frames.

Labeling (Annotating): This is the manual labor part. In Roboflow, you will draw a box around every die in every photo and label it (1, 2, 3, 4, 5, or 6).

Train in the Cloud: Use the Google Colab "YOLOv8 Training" notebook.

It’s free and uses Google's powerful computers.

You’ll point it to your Roboflow dataset.

It will output a file called best.pt. This is your "Brain" file.

Phase 3: Coding the Application

This is where you hand the requirements document to your coding assistant. You'll want to build this in "modules."

Step 1: The GUI & Calibration Module

Ask the assistant to build a screen where you can:

Click the corners of your 6 irregular scoring zones.

Assign a multiplier to each zone.

Save these coordinates to a JSON file so you don't have to do it every time you turn the app on.

Step 2: The Logic Engine (State Machine)

This part handles the "4 Major Rounds" and player order.

The Turn Tracker: Code the logic that switches from 1→10 in Round 1 and 10→1 in Round 2.

The "Stillness" Check: Tell the assistant to use a "Motion Detection" filter. The app should only update the score when the pixels stop moving for 1.5 seconds (after a flick).

Step 3: Integration (The Final Build)

Connect the "Brain" (best.pt) to the live camera feed.

Detection: The app finds a die → determines the color (Player) → determines the pips (Value).

Mapping: The app checks the (x,y) center of that die against your 6 custom polygons.

Overlay: The app draws the colored zones and a "Score Bubble" over each die.

Phase 4: Testing & "Edge Cases"

Before your first big game night, test these three scenarios:

The "Bumper" Test: Flick a die into another die. Does the score for the first die update correctly if it moves into a new zone?

The "Off-Board" Test: If a die flies off the table, ensure the "Next Player" button works to keep the game moving.

The "Hand" Test: Make sure the AI doesn't try to "score" your hand while you are flicking. (The "Stillness" check in Phase 3, Step 2 fixes this).

What you need to do right now:

Mount your camera and take a high-res photo of your board.

Download your "Flick" video to get your dataset ready.

Would you like me to provide the specific Python code for the "Motion Detection" filter so your coding assistant knows exactly how to tell when a flick has finished?

User prompt: Has any thing changed on how we will implement this? I have a usb webcam I can mount above the board. I think I would need the ability to point out the bounds of each scoring zone each game setup as the camera and board won’t be fixed each time. What else am I missing? Super ease of use and a UX that instantly clicks is what I’m looking for

Response: No fundamental principles have changed, but because the setup moves each time and you need maximum ease of use and zero friction, the implementation should be streamlined to eliminate the biggest headache: manually re-clicking 6 complex polygons every single time you sit down to play.

Here are the crucial workflow adjustments and the missing pieces needed to make the UX frictionless.

1. The Missing Piece: ArUco / Fiducial Auto-Alignment

Re-drawing 6 non-rectangular polygons by hand on every game night will quickly become tedious.

Instead of re-clicking 30+ points each setup:

Stick tiny visual marker stickers (ArUco markers / AprilTags) on the corners of your board.

Calibrate once: Click your 6 scoring zones once in high detail. The software records those zones relative to the markers.

Instant setup on Game Night: Point your USB webcam at the board. OpenCV detects the corner tags automatically in milliseconds and computes a planar homography to snap the scoring zones right onto the board.

Manual fallback: Provide a simple "Adjust Zones" mode with draggable corner handles, but 95% of the time, the marker alignment will work automatically.

2. A 3-Step Wizard for Flawless UX

For an interface that feels like a finished, consumer-grade app rather than a clunky developer script, use an onboarding wizard before dropping into the game:

[ Step 1: Camera & Alignment ] ──> [ Step 2: Player & Color Tap ] ──> [ Step 3: Game HUD ]

Step 1: Alignment & Zones:

Shows live video.

Displays your 6 zones as colored, semi-transparent overlays.

If the board shifted, either auto-aligns via markers or lets you drag points into position.

Step 2: Player Roster & One-Click Color Sampling:

Select number of players (e.g., 4).

Place one die of each player's color into a designated "sample circle" on screen and click "Sample."

The app records the color in HSV space (Hue, Saturation, Value) directly under current lighting conditions, preventing color misclassifications.

Step 3: The Live HUD:

Full-screen, clean broadcast graphic.

Giant banner showing "Whose Turn It Is" and "Die X of 4."

Live mini-leaderboard showing running round totals and locked scores.

3. State Management & Motion Gating (The "No False Triggers" Rule)

In Tumblin' Dice, hands regularly reach across the board and dice tumble rapidly across tiers. The software must know when to evaluate the board.

Motion Gate (Frame Differencing):

Compute the difference between consecutive video frames (cv2.absdiff).

While motion is detected, freeze the scoreboard and display a subtle badge: "Rolling..."

When motion settles for ≈1.2 seconds, run the YOLO pip detection and color matching.

Live HUD Overlay:

Instantly draw a bright ring around every detected die showing: [Player Name] - [Zone Value x Pip Value = Total].

If a die lands right on a boundary line, players immediately see which zone the computer awarded.

4. Essential Failsafes for Smooth Game Nights

Tabletop computer vision can encounter unexpected edge cases. Build in these quick-access controls so games never grind to a halt:

Big "Off-Board / 0 Pts" Button: If a player flicks a die completely off the table, the camera will see nothing change. A prominent spacebar tap or on-screen button marks that flick as a 0 and advances to the next player.

Click-to-Correct Override: If a die has unusual glare and reads as a 3 instead of a 5, clicking the die on screen opens a radial wheel (1–6) to correct it in two seconds.

Undo Flick: In case a die is accidentally dropped or a player shoots out of turn, a simple "Undo Last Action" restores the previous frame's scores.

Recommended App Architecture

Role

Technology

Why

Front-End / UI

PyQt6 or CustomTkinter

Native desktop window with embedded video canvas, big buttons, clean fonts, and crisp overlays.

Vision & Math

OpenCV (cv2) + NumPy

Handles marker tracking, warping, frame differencing, and pointPolygonTest.

Inference Engine

Ultralytics YOLOv8 (v8n or v8s)

Detects bounding boxes and pip counts in under 20ms per frame.

Color Classifier

HSV Distance / KNN

Matches the sampled player color to the cropped die image.

User prompt: This is great. Write this up as a spec for me to hand to a dev agent and also tell me where to get the April tags or the tags you suggest

Response: Where to Get the Visual Tags (ArUco / AprilTags)

You do not need to buy anything special—you generate and print them on standard printer paper or sticker labels.

Tag Standard: Use OpenCV ArUco markers with the DICT_4X4_50 dictionary. 4×4 markers are lower resolution than 6×6 or 7×7, making them significantly faster and more reliable for webcams to detect from overhead distances.

Online Generator: Visit an open-source generator like chev.me/arucogen or marker.rcao.ch.

Select Dictionary: 4x4 (50, 100, 250, 1000) → Pick DICT_4X4_50.

Set Marker ID: Generate 4 unique IDs (e.g., ID 0, 1, 2, 3).

Set Size: ≈30 mm to 40 mm (1.2 to 1.5 inches) square.

Placement: Print and affix the 4 stickers permanently to the 4 outer rigid corners of the wooden Tumblin' Dice board (e.g., ID 0 = Top-Left, ID 1 = Top-Right, ID 2 = Bottom-Right, ID 3 = Bottom-Left).

Generation Script (Alternative): Your developer can also generate the printable image files directly using OpenCV:import cv2aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)for marker_id in range(4):    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 300)    cv2.imwrite(f"marker_{marker_id}.png", marker_img)

Product Requirements & Technical Specification: Tumblin' Dice Live VAR Scorer

1. Executive Summary & Objective

Build a turnkey desktop application (PyQt6 + OpenCV + YOLOv8) that delivers frictionless, automated live scoring for physical games of Tumblin' Dice using an overhead USB webcam.

Because physical board setup and camera mounts vary between sessions, the system relies on ArUco fiducial marker tracking to automatically align non-rectangular scoring zones without requiring the user to redraw 6 complex polygons every time they play.

2. Core Game Rules & State Engine

Players: 2 to 10 players.

Dice Pool: Each player has 4 dice of a distinct, uniform color (up to 40 dice total on the board).

Structure: 4 Major Rounds.

Within each Major Round, each player takes 4 turns (flicks 1 die at a time).

Turn Order:

Major Round 1: Forward (P1​→Pn​)

Major Round 2: Reverse (Pn​→P1​)

Major Round 3: Forward (P1​→Pn​)

Major Round 4: Reverse (Pn​→P1​)

Score Finality: Scores for a Major Round are unlocked and dynamic until the round ends. A flicked die may knock previous dice into different zones or off the board entirely; the live score must adjust in real time after every flick. Once all players have completed their 4 flicks for the Major Round, the round score freezes, all dice are cleared, and the next Major Round begins.

Die Score Formulation: Score=Top Face Pip Value (1–6)×Zone Multiplier.

Out of Bounds / Gutter: Any die not located inside a defined scoring polygon earns 0 points.

3. UI/UX Architecture & 3-Step Setup Wizard

The app must feature a modern, high-contrast, clean broadcast layout built in PyQt6 or CustomTkinter.

[ Step 1: Camera & Alignment ] ──> [ Step 2: Player & Color Tap ] ──> [ Step 3: Live Game HUD ]

Step 1: Camera Setup & Zone Mapping

Camera Selection: Dropdown to select USB camera device index and stream resolution (default 1920×1080 @ 30 FPS).

ArUco Marker Detection:

Detect 4 corner markers (DICT_4X4_50, IDs 0, 1, 2, 3).

Display green tracking boxes over detected markers.

Zone Mapping (One-Time Setup with Profile Save):

Support up to 6 custom polygon zones.

Clicking the preview allows adding N-point irregular vertices for each zone.

Assign a multiplier value (1×,2×,3×,4×, etc.) to each zone.

Persistence: Save zone vertices relative to the 4 ArUco markers' coordinate space into zones_config.json.

Auto-Recovery: On subsequent launches, once the camera detects the 4 markers, it automatically applies a perspective homography matrix to project the 6 zones into their exact positions on the board without manual redrawing.

Provide an "Adjust Vertices" toggle with draggable handles for fine-tuning.

Step 2: Player Registration & Color Sampling

Roster Configuration: Add between 2 and 10 players with custom names.

One-Tap Color Sampling:

Place an on-screen crosshair / circle ring over the live feed.

Instruct user: "Place [Player]'s die inside the circle and click Sample."

Capture a 40×40 px region inside the die face, convert to HSV color space, compute the mean Hue, Saturation, and Value, and store color boundaries ([H_min, S_min, V_min] to [H_max, S_max, V_max]) or color cluster centroids.

Step 3: Live Game HUD & Broadcast Screen

Main Feed: High-framerate video preview displaying:

Color-coded, semi-transparent overlays for the 6 zones (opacity ≈0.25).

Circular bounding highlight around each detected die showing: [Player Name] | [Pips] x [Mult] = [Pts].

Current Turn Banner (Top/Prominent):

Displays active player name, their designated color badge, current Major Round (1/4), and die count (e.g., "Player 3 | Flick 2 of 4").

Sidebar Leaderboard:

Real-time list of all players ranked by total cumulative score.

Shows breakdowns for Rounds 1, 2, 3, 4, and Current Round tentative score.

Action Buttons & Failsafes:

"Off-Board / 0 Pts" (Spacebar shortcut): Immediately records a completed zero-point flick if a die flew completely off-camera, advancing the turn.

"Manual Edit / Override": Clicking any detected die on screen opens a quick popup (1–6) to correct a misidentified pip value.

"Undo Flick": Reverts the game state to the previous turn in case of an accidental roll or wrong player shooting.

"Next Turn / Lock Frame": Force-triggers detection if auto-motion trigger is delayed.

4. Computer Vision & Recognition Pipeline

Raw Camera Frame      │      ▼Motion Gate (Frame Differencing) ──[ Motion Active ]──► Show "Rolling..." (Hold State)      │      ▼ [ Motion Settled > 1.2s ]Detect 4 ArUco Markers ──► Compute Homography ──► Transform Zone Coordinates      │      ▼Run YOLOv8 Model ──► Locate Dice Bounding Boxes + Top Pips (Class 1 to 6)      │      ▼Crop Box ──► Mask Pips ──► Sample Base Color in HSV ──► Match to Nearest Player      │      ▼Die Centroid (x, y) ──► cv2.pointPolygonTest against 6 Zones ──► Assign Multiplier      │      ▼Update Game State & Render Live Overlays

4.1 Motion Gating & Settle Detection

To avoid misclassifying tumbling dice or moving hands, maintain an accumulation frame buffer.

Calculate absolute difference between consecutive grayscale frames:delta = cv2.absdiff(prev_gray, curr_gray)thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]movement_score = np.sum(thresh)

If movement_score > motion_threshold: flag status as "Rolling..." and pause re-scoring.

When movement_score <= motion_threshold continuously for 1.2 seconds, trigger the full vision analysis pass.

4.2 Board Alignment & Normalization

Detect ArUco corners using cv2.aruco.ArucoDetector.

Use the detected 4 marker centroids to compute a perspective transform matrix:H, _ = cv2.findHomography(stored_marker_pts, current_marker_pts)

Transform the stored zone polygon coordinates using cv2.perspectiveTransform so the zones track board position dynamically.

4.3 Die & Pip Detection

Run an inference pass using an Ultralytics YOLOv8 model (yolov8n or yolov8s) custom-trained to classify top die faces: die_1, die_2, die_3, die_4, die_5, die_6.

Filter detections with a confidence score threshold (conf≥0.65).

Compute the center coordinate (xc​,yc​) of the bounding box for spatial mapping.

4.4 Color Classification (Player Assignment)

For each detected bounding box, crop the die sub-image.

Exclude the center pips (which are typically contrasting black or white) by thresholding or sampling the outer margins of the die face.

Convert the cropped base die material to HSV color space.

Compare against the registered player color profiles using Euclidean distance in HSV or a K-Nearest Neighbors (KNN) classifier:Dist=ΔH2+ΔS2+ΔV2​

Map the die to the closest matching player profile.

4.5 Spatial Zone Mapping

Iterate through the 6 active zone polygons.

Test the die center point against each polygon:inside = cv2.pointPolygonTest(zone_polygon_pts, (x_c, y_c), measureDist=False)

If inside >= 0, assign that zone's multiplier; if no zone matches, the multiplier is 0 (Gutter/Out of Bounds).

5. Technical Stack & Deliverables

Layer

Recommended Technology

Language

Python 3.10+

UI Framework

PyQt6 (utilizing QThread for video processing to avoid UI lag)

Vision & Transforms

OpenCV (opencv-contrib-python), NumPy

Object Detection

Ultralytics YOLOv8

Configuration Storage

JSON (zones_config.json, player_profiles.json)

Development Milestones

Milestone 1: Video feed window with live ArUco detection and persistent zone drawing/transform logic.

Milestone 2: Motion gating filter and player color sampling interface.

Milestone 3: YOLO model integration, pip detection, and pointPolygonTest scoring math.

Milestone 4: Full game state machine (4 major rounds, turn sequence reversal, 10-player scoreboard, manual override modal).