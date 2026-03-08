"""
Test AgeGenderModule — age and gender estimation.
Models are auto-downloaded on first run.
Requires webcam. Press 'q' to quit.
"""

import sys
sys.path.insert(0, '..')

import cvfiq


def main():
    print("=== Age & Gender Detector Test ===")
    print("  Models auto-downloaded on first run.")
    print("  Press 'q' to quit.")

    try:
        detector = cvfiq.age_gender()
    except Exception as e:
        print(f"  AgeGender module not available: {e}")
        return

    with cvfiq.Camera(0, showFPS=True, title="Age Gender Test") as cam:
        for img in cam:
            results, img = detector.findAgeGender(img)
            for r in results:
                print(f"  {r['gender']} ({r['genderConf']:.0%}), age {r['age']}")
            cam.show(img)

    print("  PASSED")


if __name__ == "__main__":
    main()
