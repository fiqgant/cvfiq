"""
Test PlotModule.
No camera needed — uses blank canvas.
"""

import sys
import cv2
import numpy as np
sys.path.insert(0, '..')

from cvfiq.PlotModule import LivePlot

def main():
    print("=== PlotModule Test ===")

    plot = LivePlot(w=640, h=480, yLimit=[0, 100])
    print(f"  Created LivePlot(640x480, yLimit=[0,100])")

    canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    # Feed values and render
    for val in [10, 30, 50, 70, 90, 60, 40, 20]:
        out = plot.update(val, canvas.copy())

    assert out.shape == (480, 640, 3), f"Unexpected shape: {out.shape}"
    print(f"  Output shape: {out.shape}")

    # Show result (press any key to continue)
    cv2.imshow("PlotModule Test — press any key", out)
    print("  Window open. Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("  PASSED")

if __name__ == "__main__":
    main()
