"""
Test PlotModule.
No camera needed — shows a sine wave plot.
Press any key to close.
"""

import sys, math
sys.path.insert(0, '..')

import cvfiq

def main():
    print("=== PlotModule Test ===")

    plotter = cvfiq.plot(w=640, h=480, yLimit=[-100, 100])
    x = 0

    print("  Sine wave running. Press any key to close.")
    for _ in range(200):
        x = (x + 2) % 360
        val = int(math.sin(math.radians(x)) * 100)
        imgPlot = plotter.update(val, color=(0, 200, 255))
        cvfiq.imshow("PlotModule Test — press any key", imgPlot)
        if cvfiq.waitKey(10) != -1:
            break

    cvfiq.destroyAllWindows()
    print("  PASSED")

if __name__ == "__main__":
    main()
