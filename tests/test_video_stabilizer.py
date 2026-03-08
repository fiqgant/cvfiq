"""
Test VideoStabilizerModule.
Shows original vs stabilized video side by side.
Controls: q=quit  r=reset  s=cycle radius  b=cycle border
"""

import sys
sys.path.insert(0, '..')

import cvfiq
import numpy as np

SMOOTH_RADII = [5, 15, 30]
BORDER_MODES = ['black', 'replicate', 'reflect']

def main():
    print("=== VideoStabilizerModule Test ===")
    print("  Move the camera around to see stabilization.")
    print("  Controls: q=quit  r=reset  s=cycle radius  b=cycle border")

    smooth_idx = 1
    border_idx = 0
    stab = cvfiq.stabilizer(smoothRadius=SMOOTH_RADII[smooth_idx],
                             border=BORDER_MODES[border_idx])

    with cvfiq.Camera(0, title="VideoStabilizer Test") as cam:
        for img in cam:
            imgStab    = stab.stabilize(img)
            smoothness = stab.getSmoothness()
            H, W       = img.shape[:2]

            cvfiq.putText(imgStab, f"Smooth: {smoothness:.3f}  R:{SMOOTH_RADII[smooth_idx]}  B:{BORDER_MODES[border_idx]}",
                          (10, 30), cvfiq.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cvfiq.putText(img, "Original", (10, 30),
                          cvfiq.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            bar_w = int(smoothness * (W - 20))
            bar_c = (0, int(255 * smoothness), int(255 * (1 - smoothness)))
            cvfiq.rectangle(imgStab, (10, H - 20), (W - 10, H - 8), (50, 50, 50), -1)
            cvfiq.rectangle(imgStab, (10, H - 20), (10 + bar_w, H - 8), bar_c, -1)

            combined = np.hstack([img, np.full((H, 4, 3), 255, dtype=np.uint8), imgStab])
            cvfiq.imshow("VideoStabilizer Test  [q=quit | r=reset | s=radius | b=border]", combined)

            key = cvfiq.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                stab.reset()
                print("  [INFO] Stabilizer reset")
            elif key == ord('s'):
                smooth_idx = (smooth_idx + 1) % len(SMOOTH_RADII)
                stab.smoothRadius = SMOOTH_RADII[smooth_idx]
                print(f"  [INFO] smoothRadius -> {SMOOTH_RADII[smooth_idx]}")
            elif key == ord('b'):
                border_idx = (border_idx + 1) % len(BORDER_MODES)
                stab.border = BORDER_MODES[border_idx]
                print(f"  [INFO] border -> {BORDER_MODES[border_idx]}")

    print("  PASSED")

if __name__ == "__main__":
    main()
