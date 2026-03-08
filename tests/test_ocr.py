"""
Test OCRModule — optical character recognition.
Requires: pip install pytesseract + tesseract binary installed.
Requires webcam. Show text to the camera. Press 'q' to quit.
"""

import sys
sys.path.insert(0, '..')

import cvfiq

print("=== OCR Reader Test ===")
print("  Requires: pip install pytesseract + tesseract installed.")
print("  Show printed text to the camera.")
print("  Press 'q' to quit.")

try:
    reader = cvfiq.ocr()
except Exception as e:
    print(f"  OCR not available: {e}")
    sys.exit(0)

with cvfiq.Camera(0, showFPS=True, title="OCR Test") as cam:
    for img in cam:
        texts, img = reader.findText(img)
        for t in texts:
            print(f"  [{t['confidence']:.0f}%] {t['text']}")
        cam.show(img)
