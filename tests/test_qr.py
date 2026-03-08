"""
Test QRModule — QR code and barcode detection.
Requires webcam. Show a QR code to the camera.
Press 'q' to quit.
"""

import sys
sys.path.insert(0, '..')

import cvfiq


def main():
    print("=== QR / Barcode Detector Test ===")
    print("  Show a QR code or barcode to the camera.")
    print("  Press 'q' to quit.")

    scanner = cvfiq.qr()

    with cvfiq.Camera(0, showFPS=True, title="QR Test") as cam:
        for img in cam:
            codes, img = scanner.findCodes(img)
            for c in codes:
                print(f"  Detected [{c['type']}]: {c['data']}")
            cam.show(img)

    print("  PASSED")


if __name__ == "__main__":
    main()
