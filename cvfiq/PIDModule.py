import cv2
import numpy as np
import time


class PID:
    def __init__(self, pidVals, targetVal, axis=0, limit=None, iLimit=None):
        """
        :param pidVals: [Kp, Ki, Kd]
        :param targetVal: Target value to reach
        :param axis: 0 = x-axis, 1 = y-axis (for draw())
        :param limit: [min, max] output clamp
        :param iLimit: [min, max] integral windup clamp
        """
        self.pidVals = pidVals
        self.targetVal = targetVal
        self.axis = axis
        self.pError = 0
        self.limit = limit
        self.iLimit = iLimit
        self.I = 0
        self.pTime = time.time()

    def update(self, cVal):
        # Current Value - Target Value
        t = time.time() - self.pTime
        if t == 0:
            t = 1e-6
        error = cVal - self.targetVal
        P = self.pidVals[0] * error
        self.I = self.I + (self.pidVals[1] * error * t)

        # Integral windup clamp
        if self.iLimit is not None:
            self.I = float(np.clip(self.I, self.iLimit[0], self.iLimit[1]))

        D = (self.pidVals[2] * (error - self.pError)) / t

        result = P + self.I + D

        if self.limit is not None:
            result = float(np.clip(result, self.limit[0], self.limit[1]))
        self.pError = error
        self.pTime = time.time()

        return result

    def reset(self):
        """Reset PID state (integral and previous error)."""
        self.I = 0
        self.pError = 0
        self.pTime = time.time()

    def draw(self, img, cVal):
        h, w, _ = img.shape
        if self.axis == 0:
            cv2.line(img, (self.targetVal, 0), (self.targetVal, h), (255, 0, 255), 1)
            cv2.line(img, (self.targetVal, cVal[1]), (cVal[0], cVal[1]), (255, 0, 255), 1, 0)
        else:
            cv2.line(img, (0, self.targetVal), (w, self.targetVal), (255, 0, 255), 1)
            cv2.line(img, (cVal[0], self.targetVal), (cVal[0], cVal[1]), (255, 0, 255), 1, 0)

        cv2.circle(img, tuple(cVal), 5, (255, 0, 255), cv2.FILLED)

        return img


def main():
    from cvfiq.FaceDetectionModule import FaceDetector
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()
    # For a 640x480 image center target is 320 and 240
    xPID = PID([1, 0.000000000001, 1], 640 // 2)
    yPID = PID([1, 0.000000000001, 1], 480 // 2, axis=1, limit=[-100, 100])

    while True:
        success, img = cap.read()
        img, bboxs = detector.findFaces(img)
        if bboxs:
            x, y, w, h = bboxs[0]["bbox"]
            cx, cy = bboxs[0]["center"]
            xVal = int(xPID.update(cx))
            yVal = int(yPID.update(cy))

            xPID.draw(img, [cx, cy])
            yPID.draw(img, [cx, cy])

            cv2.putText(img, f'x:{xVal} , y:{yVal} ', (x, y - 100), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
