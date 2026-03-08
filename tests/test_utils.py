"""
Test Utils module.
No camera needed — tests all utility functions on blank/test images.
Press any key to close windows.
"""

import sys
import cv2
import numpy as np
sys.path.insert(0, '..')

from cvfiq.Utils import (
    stackImages, putTextRect, cornerRect, overlayPNG, rotateImage, findContours
)

def main():
    print("=== Utils Module Test ===")

    img1 = np.zeros((300, 400, 3), dtype=np.uint8)
    img2 = np.ones((300, 400, 3), dtype=np.uint8) * 128
    img3 = np.zeros((300, 400, 3), dtype=np.uint8)
    img3[:] = (50, 100, 200)
    img4 = np.zeros((300, 400, 3), dtype=np.uint8)
    img4[:] = (200, 50, 100)

    # stackImages
    stacked = stackImages([img1, img2, img3, img4], cols=2, scale=0.5)
    assert stacked.shape[2] == 3, "stackImages failed"
    print(f"  stackImages OK: {stacked.shape}")

    # putTextRect
    canvas = img1.copy()
    canvas, box = putTextRect(canvas, "Hello cvfiq", (50, 50),
                               scale=1.5, thickness=2, colorR=(0, 200, 0))
    print(f"  putTextRect OK, bbox={box}")

    # cornerRect
    canvas2 = img2.copy()
    cornerRect(canvas2, (50, 50, 200, 150), l=20, t=3, rt=1)
    print(f"  cornerRect OK")

    # rotateImage
    rotated = rotateImage(img3.copy(), 45)
    assert rotated.shape == img3.shape, "rotateImage shape mismatch"
    print(f"  rotateImage OK: {rotated.shape}")

    # findContours
    mask = np.zeros((300, 400), dtype=np.uint8)
    cv2.rectangle(mask, (50, 50), (150, 150), 255, -1)
    cv2.rectangle(mask, (200, 100), (350, 250), 255, -1)
    contours, hierarchy = findContours(img1.copy(), mask, minArea=500)
    print(f"  findContours OK: found {len(contours)} contours")
    assert len(contours) == 2, f"Expected 2 contours, got {len(contours)}"

    # Show results
    combined = stackImages([stacked, canvas, canvas2, rotated], cols=2, scale=0.7)
    cv2.imshow("Utils Test — press any key", combined)
    print("  Window open. Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("  PASSED")

if __name__ == "__main__":
    main()
