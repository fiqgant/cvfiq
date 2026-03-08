"""
Test Utils module.
No camera needed — tests all utility functions on synthetic images.
Press any key to close.
"""

import sys
sys.path.insert(0, '..')

import cvfiq
import numpy as np

def main():
    print("=== Utils Module Test ===")

    img1 = np.zeros((300, 400, 3), dtype=np.uint8)
    img2 = np.full((300, 400, 3), 128, dtype=np.uint8)
    img3 = np.full((300, 400, 3), (50, 100, 200), dtype=np.uint8)
    img4 = np.full((300, 400, 3), (200, 50, 100), dtype=np.uint8)

    # stackImages
    stacked = cvfiq.stackImages([img1, img2, img3, img4], cols=2, scale=0.5)
    assert stacked.shape[2] == 3
    print(f"  stackImages OK: {stacked.shape}")

    # putTextRect
    canvas, box = cvfiq.putTextRect(img1.copy(), "Hello cvfiq", (50, 50),
                                     scale=1.5, thickness=2, colorR=(0, 200, 0))
    print(f"  putTextRect OK, bbox={box}")

    # cornerRect
    cvfiq.cornerRect(img2.copy(), (50, 50, 200, 150), l=20, t=3, rt=1)
    print("  cornerRect OK")

    # rotateImage
    rotated = cvfiq.rotateImage(img3.copy(), 45)
    assert rotated.shape == img3.shape
    print(f"  rotateImage OK: {rotated.shape}")

    # findContours
    mask = np.zeros((300, 400), dtype=np.uint8)
    cvfiq.rectangle(mask, (50, 50), (150, 150), 255, -1)
    cvfiq.rectangle(mask, (200, 100), (350, 250), 255, -1)
    contours, _ = cvfiq.findContours(img1.copy(), mask, minArea=500)
    print(f"  findContours OK: found {len(contours)} contours")
    assert len(contours) == 2, f"Expected 2, got {len(contours)}"

    combined = cvfiq.stackImages([stacked, canvas, rotated], cols=3, scale=0.7)
    cvfiq.imshow("Utils Test — press any key", combined)
    cvfiq.waitKey(0)
    cvfiq.destroyAllWindows()

    print("  PASSED")

if __name__ == "__main__":
    main()
