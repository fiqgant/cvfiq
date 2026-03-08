"""
Run all non-interactive tests (no camera, no GUI).
Camera-based tests must be run individually.
"""

import sys
import traceback
sys.path.insert(0, '..')

TESTS = [
    ("FPS",        "test_fps",   "main"),
    ("PIDModule",  "test_pid",   "main"),
    ("Utils",      "test_utils", None),   # has GUI, skip in batch
]

# Non-interactive logic-only tests
def run_fps():
    import time
    from cvfiq.FPS import FPS
    fps = FPS(avgCount=10)
    for _ in range(15):
        time.sleep(0.033)
        f = fps.update()
    assert 25 < f < 45, f"FPS out of range: {f}"
    return True

def run_pid():
    from cvfiq.PIDModule import PID
    pid = PID([0.5, 0.01, 0.1], targetVal=100)
    out = pid.update(80)
    assert isinstance(out, (int, float)), "PID output not a number"
    pid.reset()

    pid2 = PID([0.1, 1.0, 0.0], targetVal=100, iLimit=[-50, 50])
    for _ in range(100):
        pid2.update(0)
    assert abs(pid2.I) <= 50, f"iLimit failed: {pid2.I}"

    pid3 = PID([0.5, 0.0, 0.0], targetVal=0, limit=[-100, 100])
    out3 = pid3.update(500)
    assert -100 <= out3 <= 100, f"limit failed: {out3}"
    return True

def run_plot():
    import numpy as np
    from cvfiq.PlotModule import LivePlot
    plot = LivePlot(w=640, h=480, yLimit=[0, 100])
    import cv2
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    for val in [10, 50, 90, 30, 70]:
        out = plot.update(val, canvas.copy())
    assert out.shape == (480, 640, 3)
    return True

def run_color():
    import numpy as np
    from cvfiq.ColorModule import ColorFinder
    cf = ColorFinder(trackBar=False)
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[80:120, 80:120] = [0, 200, 0]
    hsvVals = {'hmin': 35, 'smin': 80, 'vmin': 80,
               'hmax': 85, 'smax': 255, 'vmax': 255}
    imgColor, mask = cf.update(img, hsvVals)
    assert mask.sum() > 0, "Mask is empty"
    return True

def run_aruco():
    import numpy as np
    import cv2
    from cvfiq.ArucoModule import ArucoDetector
    det = ArucoDetector(dictType='DICT_4X4_50')
    marker = det.generateMarker(1, size=200)
    assert marker.shape == (200, 200), f"Marker shape: {marker.shape}"

    # Detect from blank (no markers — should return empty)
    blank = np.zeros((300, 300, 3), dtype=np.uint8)
    markers, _ = det.findMarkers(blank)
    assert isinstance(markers, list)
    return True

def run_video_stabilizer():
    import numpy as np
    from cvfiq.VideoStabilizerModule import VideoStabilizer
    vs = VideoStabilizer(smoothRadius=5)
    for _ in range(10):
        frame = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
        out = vs.stabilize(frame)
        assert out.shape == frame.shape
    s = vs.getSmoothness()
    assert 0.0 <= s <= 1.0, f"Smoothness out of range: {s}"
    vs.reset()
    return True

def run_imports():
    """Just verify all modules import cleanly."""
    modules = [
        'cvfiq.FPS', 'cvfiq.PIDModule', 'cvfiq.PlotModule',
        'cvfiq.ColorModule', 'cvfiq.HandTrackingModule',
        'cvfiq.FaceDetectionModule', 'cvfiq.FaceMeshModule',
        'cvfiq.PoseModule', 'cvfiq.SelfiSegmentationModule',
        'cvfiq.ClassificationModule', 'cvfiq.SerialModule',
        'cvfiq.ArucoModule', 'cvfiq.DNNModule',
        'cvfiq.VideoStabilizerModule', 'cvfiq.GestureModule',
        'cvfiq.FaceLandmarkerModule', 'cvfiq.ObjectDetectorModule',
        'cvfiq.Utils',
    ]
    for m in modules:
        __import__(m)
    return True


BATCH_TESTS = [
    ("All imports",       run_imports),
    ("FPS",               run_fps),
    ("PIDModule",         run_pid),
    ("PlotModule",        run_plot),
    ("ColorModule",       run_color),
    ("ArucoModule",       run_aruco),
    ("VideoStabilizer",   run_video_stabilizer),
]

if __name__ == "__main__":
    print("=" * 50)
    print("  cvfiq — Batch Test Runner")
    print("  (no camera, no GUI required)")
    print("=" * 50)

    passed = 0
    failed = 0

    for name, fn in BATCH_TESTS:
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            traceback.print_exc()
            failed += 1

    print("=" * 50)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 50)

    if failed:
        print()
        print("Camera-based tests (run individually):")

    camera_tests = [
        ("test_color.py",              "ColorModule webcam"),
        ("test_hand_tracking.py",      "HandTrackingModule webcam"),
        ("test_face_detection.py",     "FaceDetectionModule webcam"),
        ("test_face_mesh.py",          "FaceMeshModule webcam"),
        ("test_pose.py",               "PoseModule webcam"),
        ("test_selfie_segmentation.py","SelfiSegmentation webcam"),
        ("test_aruco.py",              "ArucoModule webcam"),
        ("test_video_stabilizer.py",   "VideoStabilizer webcam"),
        ("test_plot.py",               "PlotModule GUI"),
        ("test_utils.py",              "Utils GUI"),
        ("test_gesture.py",            "GestureModule (needs .task model)"),
        ("test_face_landmarker.py",    "FaceLandmarkerModule (needs .task model)"),
        ("test_object_detector.py",    "ObjectDetectorModule (needs .tflite model)"),
    ]

    print()
    print("  Camera / GUI tests — run individually:")
    for fname, desc in camera_tests:
        print(f"    python {fname}   # {desc}")

    sys.exit(0 if failed == 0 else 1)
