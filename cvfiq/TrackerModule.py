"""
TrackerModule — Object tracking using OpenCV built-in trackers.
No extra dependencies required.
"""

import cv2


def _make_tracker(algo):
    algo = algo.upper()
    creators = {
        'CSRT':     lambda: cv2.TrackerCSRT_create(),
        'KCF':      lambda: cv2.TrackerKCF_create(),
        'MIL':      lambda: cv2.TrackerMIL_create(),
    }
    # Legacy trackers (may not exist in all builds)
    for name, attr in [('MOSSE', 'TrackerMOSSE_create'),
                        ('BOOSTING', 'TrackerBoosting_create'),
                        ('TLD', 'TrackerTLD_create'),
                        ('MEDIANFLOW', 'TrackerMedianFlow_create')]:
        try:
            fn = getattr(cv2.legacy, attr)
            creators[name] = fn
        except AttributeError:
            pass
    creator = creators.get(algo, creators['CSRT'])
    return creator()


class ObjectTracker:
    """
    Track a single object using OpenCV trackers.

    Supported algorithms: CSRT (default), KCF, MIL, MOSSE, BOOSTING, TLD, MEDIANFLOW

    Usage:
        tracker = ObjectTracker(algo='CSRT')

        # Option A: let user draw ROI with mouse
        with cvfiq.Camera(0) as cam:
            for img in cam:
                if not tracker.initialized:
                    tracker.select(img)  # opens ROI window
                success, bbox, img = tracker.update(img)
                cam.show(img)

        # Option B: provide bbox manually
        tracker.init(first_frame, (x, y, w, h))
    """

    def __init__(self, algo='CSRT'):
        self.algo         = algo.upper()
        self._tracker     = None
        self.initialized  = False

    def init(self, img, bbox):
        """Initialize tracker with bbox = (x, y, w, h)."""
        self._tracker    = _make_tracker(self.algo)
        self._tracker.init(img, tuple(int(v) for v in bbox))
        self.initialized = True
        return self

    def select(self, img, winName="Select object — Enter=confirm  C=cancel"):
        """Open ROI selector window. Returns bbox (x,y,w,h) or None."""
        bbox = cv2.selectROI(winName, img, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(winName)
        if bbox[2] > 0 and bbox[3] > 0:
            self.init(img, bbox)
            return bbox
        return None

    def update(self, img, draw=True):
        """
        Update tracker for current frame.

        Returns:
            success (bool): whether object was found
            bbox (tuple|None): (x, y, w, h) or None if lost
            img: annotated image
        """
        if not self.initialized or self._tracker is None:
            cv2.putText(img, f"[{self.algo}] Not initialized",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return False, None, img

        success, box = self._tracker.update(img)
        if success:
            x, y, w, h = [int(v) for v in box]
            if draw:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, self.algo, (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            return True, (x, y, w, h), img
        else:
            if draw:
                cv2.putText(img, f"[{self.algo}] Tracking lost — press R to reset",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return False, None, img

    def reset(self):
        """Reset tracker state."""
        self._tracker    = None
        self.initialized = False
