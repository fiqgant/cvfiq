"""
Test PIDModule.
No camera needed — pure math test.
"""

import sys
sys.path.insert(0, '..')

import cvfiq

def main():
    print("=== PID Module Test ===")

    pid = cvfiq.pid([0.5, 0.01, 0.1], targetVal=100)
    print(f"  update(80)  -> {pid.update(80):.4f}")
    print(f"  update(100) -> {pid.update(100):.4f}")
    pid.reset()
    print("  reset() OK")

    pid2 = cvfiq.pid([0.1, 1.0, 0.0], targetVal=100, iLimit=[-50, 50])
    for _ in range(100):
        pid2.update(0)
    print(f"  I after 100 iters (iLimit ±50): {pid2.I:.2f}")
    assert abs(pid2.I) <= 50, f"iLimit not working: {pid2.I}"

    pid3 = cvfiq.pid([0.5, 0.0, 0.0], targetVal=0, limit=[-100, 100])
    out3 = pid3.update(500)
    print(f"  update(500) with limit=[-100,100] -> {out3:.2f} (clamped)")
    assert -100 <= out3 <= 100, f"limit not working: {out3}"

    print("  PASSED")

if __name__ == "__main__":
    main()
