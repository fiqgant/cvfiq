"""
Test PIDModule.
No camera needed — pure math test.
"""

import sys
sys.path.insert(0, '..')

from cvfiq.PIDModule import PID

def main():
    print("=== PID Module Test ===")

    # Basic instantiation
    pid = PID([0.5, 0.01, 0.1], targetVal=100)
    print(f"  Created PID, target=100")

    # Should output negative correction (current > target or < target)
    out = pid.update(80)
    print(f"  update(80) -> {out:.4f}")

    out2 = pid.update(100)
    print(f"  update(100) -> {out2:.4f}")

    # Reset
    pid.reset()
    print(f"  reset() OK")

    # Integral windup clamp
    pid2 = PID([0.1, 1.0, 0.0], targetVal=100, iLimit=[-50, 50])
    for _ in range(100):
        pid2.update(0)  # large sustained error
    print(f"  I after 100 iterations with iLimit: {pid2.I:.2f} (should be clamped to <=50)")
    assert abs(pid2.I) <= 50, f"iLimit not working: {pid2.I}"

    # With axis limit
    pid3 = PID([0.5, 0.0, 0.0], targetVal=0, limit=[-100, 100])
    out3 = pid3.update(500)
    print(f"  update(500) with limit=[-100,100] -> {out3:.2f} (clamped)")
    assert -100 <= out3 <= 100, f"limit not working: {out3}"

    print("  PASSED")

if __name__ == "__main__":
    main()
