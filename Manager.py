import threading

import cv2 as cv
import keyboard
import numpy as np

from ImageEditor import ImageEditor
from States import States
from StateProcessor import StateProcessor

from HandDetector import HandDetector
from HandRecognition import HandRecognition
from Preprocessor import Preprocessor


class Manager:
    def __init__(self):
        self.cap = cv.VideoCapture(0)
        # ip = "http://192.168.42.129:8080/video"
        # self.cap = cv.VideoCapture(ip)
        self.frameWidth = self.cap.get(cv.CAP_PROP_FRAME_WIDTH)
        self.frameHeight = self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        self.handDetector = HandDetector(self.cap)
        self.preprocessor = Preprocessor()
        self.handReco = HandRecognition()
        self.currentState = States.Choose
        self.stateProcessor = StateProcessor()

        self.image = cv.imread("home.jpg")
        self.editor = ImageEditor(self.image)

        self.reloadThread = threading.Thread(target=self.reloadImage)
        self.reloadThread.start()

    def run(self):
        while True:
            ret, frame = self.cap.read()

            frame = self.preprocessor.run(frame)
            handPixels, DFrame, handContour = self.handDetector.run(frame)

            fingersFrame, handCenterX, handCenterY, fingerCount, fingersLandMarks = \
                self.handReco.run(handPixels, frame, handContour)

            # cv.imshow("video", frame)
            # cv.imshow("hand contour", DFrame)
            # cv.imshow("hand", handPixels)

            resultFrame = self.stateProcessor.run(fingerCount, fingersFrame)

            resultImage = self.editor.run(handCenterX, handCenterY,
                                            fingerCount, fingersLandMarks,
                                            self.stateProcessor.state)

            cv.imshow("result", resultFrame)
            cv.imshow("image", resultImage)

            k = cv.waitKey(30) & 0xff
            if k == 27:
                break

        cv.destroyAllWindows()
        self.cap.release()

    def reloadImage(self):
        while True:
            keyboard.wait("r")
            self.image = cv.imread("home.jpg")
            self.editor.image = self.image
            self.editor.findBigImage()
