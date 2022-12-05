import cv2 as cv
import numpy as np
from HandDetector import HandDetector
from Preprocessor import Preprocessor

class Manager:
    def __init__(self):
        self.cap = cv.VideoCapture(0)
        self.frameWidth = self.cap.get(cv. CAP_PROP_FRAME_WIDTH)
        self.frameHeight = self.cap.get(cv. CAP_PROP_FRAME_HEIGHT)
        self.handDetector = HandDetector(self.cap)
        self.preprocessor = Preprocessor()

    def run(self):

        while True:
            ret, frame = self.cap.read()

            frame = self.preprocessor.run(frame)
            handFrame, editedVideoFrame = self.handDetector.run(frame)
            cv.imshow("video", editedVideoFrame)
            cv.imshow("hand detection", handFrame)

            k = cv.waitKey(30) & 0xff
            if k == 27:
                break

