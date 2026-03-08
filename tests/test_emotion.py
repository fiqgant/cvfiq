"""
Test EmotionModule — face emotion detection.
Requires: pip install deepface
Requires webcam. Press 'q' to quit.
"""

import sys
sys.path.insert(0, '..')

import cvfiq


def main():
    print("=== Emotion Detector Test ===")
    print("  Requires: pip install deepface")
    print("  Press 'q' to quit.")

    try:
        detector = cvfiq.emotion()
    except Exception as e:
        print(f"  Emotion module not available: {e}")
        print("  Install: pip install deepface")
        return

    with cvfiq.Camera(0, showFPS=True, title="Emotion Test") as cam:
        for img in cam:
            emotions, img = detector.findEmotions(img)
            for e in emotions:
                print(f"  {e['emotion']}")
            cam.show(img)

    print("  PASSED")


if __name__ == "__main__":
    main()
