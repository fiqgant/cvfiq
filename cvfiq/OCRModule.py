"""
OCRModule — Text detection and recognition.
Requires: pip install pytesseract
          + tesseract-ocr installed on system:
            macOS:  brew install tesseract
            Ubuntu: sudo apt install tesseract-ocr
            Windows: https://github.com/UB-Mannheim/tesseract/wiki
"""

import cv2
import numpy as np


class OCRReader:
    """
    Read text from images using Tesseract OCR.

    Usage:
        ocr = OCRReader()
        texts, img = ocr.find(img)
        for t in texts:
            print(t["text"], t["confidence"])

    If pytesseract is not installed, find() returns ([], img) with a warning overlay.
    """

    def __init__(self, lang='eng', minConf=60, psm=11):
        """
        :param lang:    Tesseract language code (e.g. 'eng', 'ind', 'chi_sim')
        :param minConf: Minimum confidence (0-100) to include a word
        :param psm:     Tesseract page segmentation mode (11 = sparse text, no OSD)
        """
        self.lang    = lang
        self.minConf = minConf
        self.psm     = psm
        self._tess   = None
        try:
            import pytesseract
            self._tess = pytesseract
        except ImportError:
            pass

    @property
    def available(self):
        """True if pytesseract is installed and usable."""
        return self._tess is not None

    def find(self, img, draw=True):
        """
        Find and read text in img.

        Returns:
            texts (list): [{"text": str, "confidence": float, "bbox": (x,y,w,h), "center": (cx,cy)}]
            img: annotated image
        """
        if self._tess is None:
            if draw:
                cv2.putText(img, "OCR: pip install pytesseract", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return [], img

        config  = f"--psm {self.psm}"
        rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            data = self._tess.image_to_data(
                rgb, lang=self.lang, config=config,
                output_type=self._tess.Output.DICT
            )
        except Exception as e:
            if draw:
                cv2.putText(img, f"OCR error: {e}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return [], img

        results = []
        for i in range(len(data['text'])):
            text = str(data['text'][i]).strip()
            try:
                conf = float(data['conf'][i])
            except (ValueError, TypeError):
                conf = -1
            if not text or conf < self.minConf:
                continue
            x = int(data['left'][i])
            y = int(data['top'][i])
            w = int(data['width'][i])
            h = int(data['height'][i])
            cx, cy = x + w // 2, y + h // 2
            results.append({
                "text":       text,
                "confidence": conf,
                "bbox":       (x, y, w, h),
                "center":     (cx, cy),
            })
            if draw:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 200, 255), 2)
                cv2.putText(img, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

        return results, img

    def readText(self, img):
        """Return all detected text as a single string."""
        if self._tess is None:
            return ""
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            return self._tess.image_to_string(rgb, lang=self.lang).strip()
        except Exception:
            return ""
