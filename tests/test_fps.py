"""
Test FPS module.
No camera needed — pure timing test.
"""

import sys, time
sys.path.insert(0, '..')

import cvfiq

def main():
    print("=== FPS Module Test ===")

    fpsReader = cvfiq.fps(avgCount=10)

    readings = []
    for _ in range(20):
        time.sleep(0.033)   # simulate ~30 fps
        f = fpsReader.update()
        readings.append(f)

    avg = sum(readings[-10:]) / 10
    print(f"  Simulated ~30fps, measured avg: {avg:.1f} fps")
    assert 25 < avg < 35, f"Expected ~30fps, got {avg:.1f}"

    print("  PASSED")

if __name__ == "__main__":
    main()
