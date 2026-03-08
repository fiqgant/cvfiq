"""
Test MotionModule — background subtraction motion detection.
Requires webcam. Move in front of the camera.
Press 'q' to quit.
"""

import sys
sys.path.insert(0, '..')

import cvfiq

print("=== Motion Detector Test ===")
print("  Move in front of the camera to trigger motion.")
print("  Press 'q' to quit.")

detector = cvfiq.motion(minArea=500, history=200)

with cvfiq.Camera(0, showFPS=True, title="Motion Test") as cam:
    for img in cam:
        detected, regions, img = detector.findMotion(img)
        status = f"Motion: {'YES' if detected else 'no'}"
        if detected:
            status += f" ({len(regions)} regions)"
        cvfiq.text(img, status, (10, 30), color=(0, 0, 255) if detected else (0, 255, 0))
        cam.show(img)
