"""
Test TrackerModule — single-object tracking.
Requires webcam. Press 's' to select ROI to track. Press 'q' to quit.
"""

import sys
sys.path.insert(0, '..')

import cvfiq


def main():
    print("=== Object Tracker Test ===")
    print("  Press 's' to select an object to track.")
    print("  Press 'q' to quit.")

    tracker = cvfiq.tracker(algo='CSRT')
    tracking = False

    with cvfiq.Camera(0, showFPS=True, title="Tracker Test") as cam:
        for img in cam:
            if tracking:
                success, bbox, img = tracker.update(img)
                if success:
                    cvfiq.putText(img, "Tracking", (10, 30),
                                  cvfiq.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    cvfiq.putText(img, "Lost — press 's' to reselect", (10, 30),
                                  cvfiq.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    tracking = False
            else:
                cvfiq.putText(img, "Press 's' to select ROI", (10, 30),
                              cvfiq.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            key = cam.show(img)
            if key == ord('s'):
                bbox = tracker.select(img)
                if bbox:
                    tracking = True

    print("  PASSED")


if __name__ == "__main__":
    main()
