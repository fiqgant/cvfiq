"""
Test DepthModule — monocular depth estimation using MiDaS.
Model is auto-downloaded on first run.
Requires webcam. Press 'q' to quit.
"""

import sys
sys.path.insert(0, '..')

import cvfiq

print("=== Depth Estimator Test ===")
print("  Model auto-downloaded on first run.")
print("  Press 'q' to quit.")

try:
    estimator = cvfiq.depth()
except Exception as e:
    print(f"  Depth module not available: {e}")
    sys.exit(0)

with cvfiq.Camera(0, showFPS=True, title="Depth Test") as cam:
    for img in cam:
        depthMap, colorized = estimator.find(img)
        stacked = cvfiq.stackImages([img, colorized], cols=2, scale=0.8)
        cam.show(stacked)
