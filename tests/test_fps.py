"""
Test FPS module.
No camera needed — pure timing test.
"""

import sys
import time
sys.path.insert(0, '..')

from cvfiq.FPS import FPS

def main():
    print("=== FPS Module Test ===")

    fps = FPS(avgCount=10)
    print(f"  Created FPS(avgCount=10)")

    readings = []
    for i in range(20):
        time.sleep(0.033)  # simulate ~30fps
        f = fps.update()
        readings.append(f)

    avg = sum(readings[-10:]) / 10
    print(f"  Simulated ~30fps, measured avg: {avg:.1f} fps")
    assert 25 < avg < 35, f"Expected ~30fps, got {avg:.1f}"

    fps2 = FPS(avgCount=5)
    for _ in range(5):
        time.sleep(0.016)  # simulate ~60fps
    f2 = fps2.update()
    print(f"  Simulated ~60fps, last reading: {f2:.1f} fps")

    print("  PASSED")

if __name__ == "__main__":
    main()
