"""
Test EmotionModule — face emotion detection.
Requires: pip install deepface
Requires webcam. Press 'q' to quit.
"""

import sys
sys.path.insert(0, '..')

import cvfiq

print("=== Emotion Detector Test ===")
print("  Requires: pip install deepface")
print("  Press 'q' to quit.")

try:
    detector = cvfiq.emotion()
except Exception as e:
    print(f"  Emotion module not available: {e}")
    print("  Install: pip install deepface")
    sys.exit(0)

with cvfiq.Camera(0, showFPS=True, title="Emotion Test") as cam:
    for img in cam:
        emotions, img = detector.find(img)
        for e in emotions:
            print(f"  {e['emotion']}")
        cam.show(img)
