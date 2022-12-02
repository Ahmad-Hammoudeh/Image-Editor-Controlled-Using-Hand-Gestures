import cv2 as cv
import numpy as np
import threading
import keyboard


class HandDetector:
    def __init__(self, frameWidth, frameHeight):
        self.frameWidth = frameWidth
        self.frameHeight = frameHeight

        self.isCalibrated = False
        self.doCalibration = False
        self.calibrationThread = threading.Thread(target=self.calibrationTrigger)
        self.calibrationThread.start()

        self.hueLow, self.hueHigh = 0, 0
        self.satLow, self.satHigh = 0, 0
        self.valueLow, self.valueHigh = 0, 0

    def run(self, originalFrame):
        frame = np.copy(originalFrame)

        if self.doCalibration:
            self.calThreshold(originalFrame)
            self.doCalibration = False
            self.isCalibrated = True

        skinMask = self.generateSkinMask(frame)

        if self.isCalibrated:
            pass
        else:
            self.drawSamplingRecs(frame)

        return frame, skinMask

    def generateSkinMask(self, frame):
        hsvFrame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        mask = cv.inRange(hsvFrame, (self.hueLow, self.satLow, self.valueLow),
                          (self.hueHigh, self.satHigh, self.valueHigh))

        # Opening using 'ellipse shape' array
        ellipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, ellipse)

        # Dilation
        kernel = np.ones((5, 5), np.uint8)
        mask = cv.dilate(mask, kernel, iterations=1)

        return mask

    def calibrationTrigger(self):
        keyboard.wait("c")
        self.doCalibration = True

    def calThreshold(self, frame):
        samplingRec1 = frame[int(self.frameHeight * 0.35):int(self.frameHeight * 0.35) + 20,
                       int(self.frameWidth * 0.75):int(self.frameWidth * 0.75) + 20]

        # Convert to HSV
        samplingRec1 = cv.cvtColor(samplingRec1, cv.COLOR_BGR2HSV)

        samplingRec2 = frame[int(self.frameHeight * 0.50):int(self.frameHeight * 0.50) + 20,
                       int(self.frameWidth * 0.75):int(self.frameWidth * 0.75) + 20]

        samplingRec2 = cv.cvtColor(samplingRec2, cv.COLOR_BGR2HSV)

        offsetLowThreshold = 25
        offsetHighThreshold = 30
        # offsetLowThreshold = 80
        # offsetHighThreshold = 30

        meanRec1 = np.average(samplingRec1, axis=(0, 1))
        meanRec2 = np.average(samplingRec2, axis=(0, 1))

        self.hueLow = min(meanRec1[0], meanRec2[0]) - offsetLowThreshold
        self.hueHigh = max(meanRec1[0], meanRec2[0]) + offsetHighThreshold

        self.satLow = min(meanRec1[1], meanRec2[1]) - offsetLowThreshold
        self.satHigh = max(meanRec1[1], meanRec2[1]) + offsetHighThreshold

        self.valueLow = min(meanRec1[2], meanRec2[2]) - offsetLowThreshold
        self.valueHigh = max(meanRec1[2], meanRec2[2]) + offsetHighThreshold

        print(self.hueLow, self.hueHigh)
        print(self.satLow, self.satHigh)
        print(self.valueLow, self.valueHigh)

    def drawSamplingRecs(self, frame):
        start_point1, end_point1, start_point2, end_point2 = self.samplingRecPoints()

        frame = cv.rectangle(frame, start_point1, end_point1, (255, 0, 0), 2)
        frame = cv.rectangle(frame, start_point2, end_point2, (255, 0, 0), 2)

    def samplingRecPoints(self):
        start_point1 = (int(self.frameWidth * 0.75), int(self.frameHeight * 0.35))
        end_point1 = (int(self.frameWidth * 0.75) + 20, int(self.frameHeight * 0.35) + 20)

        start_point2 = (int(self.frameWidth * 0.75), int(self.frameHeight * 0.50))
        end_point2 = (int(self.frameWidth * 0.75) + 20, int(self.frameHeight * 0.50) + 20)

        return start_point1, end_point1, start_point2, end_point2
