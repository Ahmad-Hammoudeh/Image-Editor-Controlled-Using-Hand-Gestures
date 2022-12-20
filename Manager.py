import cv2 as cv
import numpy as np
from HandDetector import HandDetector
from HandRecognition import HandRecognition
from Preprocessor import Preprocessor

class Manager:
    def __init__(self):
        self.cap = cv.VideoCapture(0)
        self.frameWidth = self.cap.get(cv. CAP_PROP_FRAME_WIDTH)
        self.frameHeight = self.cap.get(cv. CAP_PROP_FRAME_HEIGHT)
        self.handDetector = HandDetector(self.cap)
        self.preprocessor = Preprocessor()
        self.handReco = HandRecognition()

    def run(self):

        while True:
            ret, frame = self.cap.read()

            frame = self.preprocessor.run(frame)
            handPixels, DFrame, handContour = self.handDetector.run(frame)

            handCenterX, handCenterY, fingerCount, fingersLandMarks =\
                self.handReco.run(handPixels, frame, handContour)

            cv.imshow("video", frame)
            cv.imshow("hand contour", DFrame)
            # cv.imshow("hand", handPixels)

            k = cv.waitKey(30) & 0xff
            if k == 27:
                break

        cv.destroyAllWindows()
        self.cap.release()