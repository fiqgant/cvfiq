"""
QRModule — QR code and barcode detection using OpenCV built-in.
No extra dependencies required.
"""

import cv2
import numpy as np


class QRDetector:
    """
    Detect QR codes and barcodes using OpenCV.

    Usage:
        qr = QRDetector()
        codes, img = qr.find(img)
        for c in codes:
            print(c["data"], c["type"], c["center"])
    """

    def __init__(self, detectBarcode=True):
        self._qr = cv2.QRCodeDetector()
        self._hasBarcode = False
        if detectBarcode:
            try:
                self._barcode = cv2.barcode.BarcodeDetector()
                self._hasBarcode = True
            except AttributeError:
                pass

    def find(self, img, draw=True):
        """
        Detect QR codes and barcodes in img.

        Returns:
            codes (list): [{"data": str, "type": str, "corners": list, "center": (cx,cy)}]
            img: annotated image
        """
        results = []

        # QR codes
        try:
            retval, decoded_info, points, _ = self._qr.detectAndDecodeMulti(img)
            if retval and points is not None:
                for i, pts in enumerate(points):
                    pts = pts.astype(int)
                    data = decoded_info[i] if decoded_info and i < len(decoded_info) else ""
                    if not data:
                        continue
                    cx = int(np.mean(pts[:, 0]))
                    cy = int(np.mean(pts[:, 1]))
                    results.append({
                        "data": data,
                        "type": "QR_CODE",
                        "corners": pts.tolist(),
                        "center": (cx, cy),
                    })
                    if draw:
                        cv2.polylines(img, [pts.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
                        cv2.putText(img, data[:30], (cx - 60, cy - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        except Exception:
            pass

        # Barcodes
        if self._hasBarcode:
            try:
                retval, decoded_info, decoded_type, points = self._barcode.detectAndDecode(img)
                if retval and points is not None:
                    pts_list = points if len(points.shape) == 3 else [points]
                    for i, pts in enumerate(pts_list):
                        pts = pts.astype(int)
                        data  = decoded_info[i]  if decoded_info  and i < len(decoded_info)  else ""
                        dtype = decoded_type[i]  if decoded_type  and i < len(decoded_type)  else "BARCODE"
                        if not data:
                            continue
                        cx = int(np.mean(pts[:, 0]))
                        cy = int(np.mean(pts[:, 1]))
                        results.append({
                            "data": data,
                            "type": str(dtype),
                            "corners": pts.tolist(),
                            "center": (cx, cy),
                        })
                        if draw:
                            cv2.polylines(img, [pts.reshape(-1, 1, 2)], True, (255, 140, 0), 2)
                            cv2.putText(img, data[:30], (cx - 60, cy - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 140, 0), 2)
            except Exception:
                pass

        return results, img
