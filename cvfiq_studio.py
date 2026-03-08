"""
cvfiq Studio — Real-time AI Vision Dashboard
Uses all cvfiq modules simultaneously in a 4-panel display.

Panels:
  Top-Left     : Pose + Face Detection + Hand Tracking + ArUco
  Top-Right    : Selfie Segmentation (cycle BG with 's')
  Bottom-Left  : Face Mesh + Blink/Mouth Detection
  Bottom-Right : Live Plot (finger count) + Color Detection + Stats

Controls:
  q     — quit
  s     — cycle background color/blur (segmentation)
  r     — reset video stabilizer
  SPACE — save screenshot
"""

import sys
import os
import cv2
import numpy as np
import time

sys.path.insert(0, os.path.dirname(__file__))

from cvfiq.FPS import FPS
from cvfiq.HandTrackingModule import HandDetector
from cvfiq.FaceDetectionModule import FaceDetector
from cvfiq.FaceMeshModule import FaceMeshDetector
from cvfiq.PoseModule import PoseDetector
from cvfiq.SelfiSegmentationModule import SelfiSegmentation
from cvfiq.ColorModule import ColorFinder
from cvfiq.ArucoModule import ArucoDetector
from cvfiq.VideoStabilizerModule import VideoStabilizer
from cvfiq.PIDModule import PID
from cvfiq.PlotModule import LivePlot
from cvfiq.Utils import putTextRect, cornerRect

# ── Optional: Gesture Module (auto-download model) ────────────────────────────
gesture_det = None
try:
    from cvfiq.GestureModule import GestureDetector

    GESTURE_MODEL = os.path.join(os.path.dirname(__file__), 'gesture_recognizer.task')
    GESTURE_URL   = ('https://storage.googleapis.com/mediapipe-models/'
                     'gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task')
    if not os.path.exists(GESTURE_MODEL):
        import urllib.request
        print(f"  [DOWNLOAD] Downloading gesture model...")
        urllib.request.urlretrieve(GESTURE_URL, GESTURE_MODEL)
        print(f"  [OK] gesture_recognizer.task downloaded")
    gesture_det = GestureDetector(GESTURE_MODEL)
    print("  [OK] GestureModule loaded")
except Exception as e:
    print(f"  [SKIP] GestureModule: {e}")

# ── Config ────────────────────────────────────────────────────────────────────
PW, PH   = 640, 360          # panel width/height (each of 4 panels)
CAM_W    = 640
CAM_H    = 480
CAM_IDX  = 0

BG_OPTIONS = [
    ('Black',   (0,   0,   0  )),
    ('Green',   (0,   200, 0  )),
    ('Blue',    (200, 0,   0  )),
    ('Red',     (0,   0,   200)),
    ('White',   (255, 255, 255)),
    ('Blur',    None           ),
]

# Orange HSV range for color detection
ORANGE_HSV = {'hmin': 5, 'smin': 100, 'vmin': 100,
               'hmax': 25, 'smax': 255, 'vmax': 255}

# ── Init modules ──────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(CAM_IDX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

fps_counter = FPS(avgCount=20)
hand_det    = HandDetector(maxHands=1, detectionCon=0.75)
face_det    = FaceDetector(minDetectionCon=0.7)
mesh_det    = FaceMeshDetector(maxFaces=1, refineLandmarks=False)
pose_det    = PoseDetector(smooth=True, smoothAlpha=0.5)
seg         = SelfiSegmentation(model=1)
color_find  = ColorFinder(trackBar=False)
aruco_det   = ArucoDetector(dictType='DICT_4X4_50')
stabilizer  = VideoStabilizer(smoothRadius=10)
xPID        = PID([0.4, 0.0001, 0.2], targetVal=CAM_W // 2, iLimit=[-300, 300])
plot        = LivePlot(w=PW, h=PH // 2, yLimit=[0, 5])

bg_idx          = 0
screenshot_count = 0
prev_gesture    = ''
gesture_timer   = 0
GESTURE_COOLDOWN = 1.5  # seconds between gesture triggers

# ── Helpers ───────────────────────────────────────────────────────────────────
def resize(img, w, h):
    return cv2.resize(img, (w, h))

def label(img, text, pos=(10, 22), scale=0.55, color=(180, 180, 180)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)

def hud(img, lines, x=10, y=45, dy=24, color=(0, 255, 120)):
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x, y + i * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)

def mood(finger_count, blink_left, blink_right, mouth_open):
    """Determine 'mood' from body signals."""
    if not blink_left and not blink_right:
        return "Sleepy", (100, 100, 255)
    if mouth_open and finger_count >= 4:
        return "Excited!", (0, 200, 255)
    if finger_count == 0:
        return "Focused", (0, 255, 200)
    if finger_count >= 3:
        return "Happy", (0, 255, 100)
    return "Neutral", (200, 200, 200)

def draw_dividers(img):
    h, w = img.shape[:2]
    cv2.line(img, (w // 2, 0),     (w // 2, h),     (60, 60, 60), 2)
    cv2.line(img, (0, h // 2),     (w, h // 2),     (60, 60, 60), 2)

print("\n=== cvfiq Studio ===")
print("  q=quit  s=cycle BG  r=reset stabilizer  SPACE=screenshot")
if gesture_det:
    print("  Gesture: Open_Palm=change BG | Thumb_Up=screenshot | Victory=reset stabilizer")

# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    success, raw = cap.read()
    if not success:
        break

    raw = cv2.flip(raw, 1)

    # ── Stabilize ─────────────────────────────────────────────────────────────
    img = stabilizer.stabilize(raw.copy())
    smoothness = stabilizer.getSmoothness()

    # ── FPS ───────────────────────────────────────────────────────────────────
    fps_val = fps_counter.update()

    # ─────────────────────────────────────────────────────────────────────────
    # PANEL 1 — Pose + Face Detection + Hand Tracking + ArUco
    # ─────────────────────────────────────────────────────────────────────────
    p1 = img.copy()

    # Pose
    p1 = pose_det.findPose(p1, draw=True)
    lmPose, _ = pose_det.findPosition(p1, draw=False)
    elbow_angle = None
    if len(lmPose) >= 17:
        try:
            elbow_angle = pose_det.findAngle(p1, 12, 14, 16)
        except Exception:
            pass

    # Face detection
    p1, bboxs = face_det.findFaces(p1, draw=True)
    face_center = bboxs[0]['center'] if bboxs else None
    face_conf   = round(bboxs[0]['score'][0], 2) if bboxs else 0

    # PID face tracker indicator
    pid_out = 0
    if face_center:
        pid_out = xPID.update(face_center[0])
        err = int(face_center[0] - CAM_W // 2)
        bar_x = CAM_W // 2 + int(np.clip(pid_out, -200, 200))
        cv2.arrowedLine(p1, (CAM_W // 2, CAM_H - 20),
                        (bar_x, CAM_H - 20), (0, 200, 255), 2, tipLength=0.3)
        cv2.putText(p1, f"PID:{pid_out:.0f}", (CAM_W // 2 - 30, CAM_H - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

    # Hand tracking
    hands, p1 = hand_det.findHands(p1, draw=True)
    finger_count = 0
    hand_type    = '-'
    if hands:
        hand         = hands[0]
        fingers      = hand_det.fingersUp(hand)
        finger_count = fingers.count(1)
        hand_type    = hand['type']

    # ArUco
    markers, p1 = aruco_det.findMarkers(p1, draw=True)

    # HUD
    hud(p1, [
        f"FPS: {fps_val:.1f}",
        f"Hand: {hand_type}  Fingers: {finger_count}",
        f"Face conf: {face_conf}",
        f"ArUco: {len(markers)} marker(s)",
        f"Elbow: {elbow_angle:.0f}°" if elbow_angle else "Elbow: -",
        f"Stabilizer: {smoothness:.2f}",
    ])
    label(p1, "POSE | FACE | HAND | ARUCO")
    cornerRect(p1, (5, 5, CAM_W - 10, CAM_H - 10), l=20, rt=0)

    # ─────────────────────────────────────────────────────────────────────────
    # PANEL 2 — Selfie Segmentation
    # ─────────────────────────────────────────────────────────────────────────
    bg_name, bg_val = BG_OPTIONS[bg_idx]
    if bg_val is None:
        imgBg = cv2.GaussianBlur(img, (55, 55), 0)
    else:
        imgBg = bg_val
    p2 = seg.removeBG(img.copy(), imgBg=imgBg, threshold=0.1, smooth=True, kernelSize=13)

    # Gesture control (with cooldown)
    gesture_name = ''
    if gesture_det:
        try:
            gestures, _ = gesture_det.findGestures(img.copy())
            if gestures:
                gesture_name = gestures[0]['gesture']
                now = time.time()
                if gesture_name != prev_gesture or (now - gesture_timer) > GESTURE_COOLDOWN:
                    if gesture_name == 'Open_Palm':
                        bg_idx = (bg_idx + 1) % len(BG_OPTIONS)
                    elif gesture_name == 'Thumb_Up':
                        fname = f"screenshot_{screenshot_count:03d}.jpg"
                        cv2.imwrite(fname, p1)
                        screenshot_count += 1
                        print(f"  [SNAP] {fname}")
                    elif gesture_name == 'Victory':
                        stabilizer.reset()
                        print("  [INFO] Stabilizer reset via gesture")
                    prev_gesture = gesture_name
                    gesture_timer = now
        except Exception:
            pass

    label(p2, f"SEGMENTATION  BG:{bg_name}  [s=cycle]")
    if gesture_name:
        cv2.putText(p2, f"Gesture: {gesture_name}", (10, CAM_H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

    # ─────────────────────────────────────────────────────────────────────────
    # PANEL 3 — Face Mesh + Blink/Mouth + Mood
    # ─────────────────────────────────────────────────────────────────────────
    p3 = img.copy()
    p3, faces = mesh_det.findFaceMesh(p3, draw=True)

    blink_l = blink_r = True
    mouth_open = False
    mar = 0.0
    if faces:
        try:
            blink  = mesh_det.blinkDetector(faces[0])
            m_info = mesh_det.mouthOpen(faces[0])
            blink_l    = blink['left']
            blink_r    = blink['right']
            mouth_open = m_info['open']
            mar        = m_info['ratio']
        except Exception:
            pass

    mood_text, mood_color = mood(finger_count, blink_l, blink_r, mouth_open)

    # Mood badge
    cv2.rectangle(p3, (CAM_W - 170, CAM_H - 45), (CAM_W - 5, CAM_H - 5), (30, 30, 30), -1)
    cv2.putText(p3, mood_text, (CAM_W - 165, CAM_H - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, mood_color, 2)

    hud(p3, [
        f"L-Eye: {'closed' if not blink_l else 'open  '}",
        f"R-Eye: {'closed' if not blink_r else 'open  '}",
        f"Mouth: {'OPEN' if mouth_open else 'closed'} (MAR {mar:.2f})",
        f"Faces: {len(faces)}",
    ], color=(255, 180, 0))
    label(p3, "FACE MESH | BLINK | MOUTH | MOOD")

    # ─────────────────────────────────────────────────────────────────────────
    # PANEL 4 — Live Plot + Color Detection + System Stats
    # ─────────────────────────────────────────────────────────────────────────
    # Live plot of finger count (top half)
    blank_plot = np.zeros((PH // 2, PW, 3), dtype=np.uint8)
    panel_plot = plot.update(finger_count, blank_plot)
    cv2.putText(panel_plot, f"Finger Count: {finger_count}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    # Color detection (bottom half — orange detection)
    imgColor, mask = color_find.update(img.copy(), ORANGE_HSV)
    mask_bgr  = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    detected  = mask.sum() > 5000

    panel_color = np.zeros((PH // 2, PW, 3), dtype=np.uint8)
    half_w = PW // 2
    panel_color[:, :half_w]  = resize(imgColor,  half_w, PH // 2)
    panel_color[:, half_w:]  = resize(mask_bgr,  half_w, PH // 2)
    cv2.putText(panel_color, "Orange detect", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    if detected:
        cv2.putText(panel_color, "DETECTED", (5, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    p4 = np.vstack([panel_plot, panel_color])
    label(p4, "LIVE PLOT | COLOR DETECT")

    # ─────────────────────────────────────────────────────────────────────────
    # Compose 2x2 grid
    # ─────────────────────────────────────────────────────────────────────────
    p1r = resize(p1, PW, PH)
    p2r = resize(p2, PW, PH)
    p3r = resize(p3, PW, PH)
    p4r = resize(p4, PW, PH)

    top    = np.hstack([p1r, p2r])
    bottom = np.hstack([p3r, p4r])
    canvas = np.vstack([top, bottom])

    draw_dividers(canvas)

    cv2.imshow("cvfiq Studio   [q=quit | s=BG | r=reset | SPACE=screenshot]", canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        bg_idx = (bg_idx + 1) % len(BG_OPTIONS)
        print(f"  BG -> {BG_OPTIONS[bg_idx][0]}")
    elif key == ord('r'):
        stabilizer.reset()
        print("  Stabilizer reset")
    elif key == ord(' '):
        fname = f"screenshot_{screenshot_count:03d}.jpg"
        cv2.imwrite(fname, canvas)
        screenshot_count += 1
        print(f"  [SNAP] {fname}")

cap.release()
cv2.destroyAllWindows()
