import logging
import math
import traceback

import numpy as np
import cv2 as cv
from math import *
from States import States


class ImageEditor:
    def __init__(self):
        self.vector = None
        self.image = None
        self.prevState = States.Choose

        self.translateX = 0
        self.translateY = 0

        self.increase = 1
        self.minScale = 0
        self.maxScale = 123
        self.isScaleInversed = True

    def run(self, handCenterX, handCenterY, fingerCount, fingersLandMarks, state):
        # cv.imshow("image", self.image)

        try:
            if state == States.Choose:
                self.prevState = States.Choose
                return self.image

            if state == States.Translate:
                self.translate(fingersLandMarks[0][0], fingersLandMarks[0][1])

            if state == States.Scale:
                self.scale(fingerCount, fingersLandMarks)

            if state == States.Rotate:
                self.rotate(
                    np.array([handCenterX, handCenterY]) -
                    np.array(fingersLandMarks[0][0], fingersLandMarks[0][1]),
                    fingerCount)
            return self.image

        except Exception as e:
            logging.error(traceback.format_exc())
            self.prevState = States.Choose
            return self.image

    def translate(self, x, y):
        if self.prevState != States.Translate:
            self.prevState = States.Translate
            self.translateX = x
            self.translateY = y
        else:
            height, width = self.image.shape[:2]

            deltaX = x - self.translateX
            deltaY = y - self.translateY

            translation_matrix = np.array(
                [[1, 0, deltaX / 5],
                 [0, 1, deltaY / 5]],
                dtype=np.float32)

            self.image = cv.warpAffine(src=self.image, M=translation_matrix, dsize=(width, height))

    def scale(self, fingerCount, fingersLandMark):
        if fingerCount == 4 and self.isScaleInversed:
            self.increase *= -1
            self.isScaleInversed = False

        if fingerCount != 2:
            return

        self.isScaleInversed = True

        p1 = fingersLandMark[0]
        p2 = fingersLandMark[1]

        distance = self.disAB(p1, p2) / 50
        # print(distance)
        scaleValue = np.interp(distance, [self.minScale, self.maxScale],
                               [0, 1])

        # print(scaleValue)
        self.image = cv.resize(self.image, (0, 0),
                               fx=1 + scaleValue * self.increase,
                               fy=1 + scaleValue * self.increase)

    def disAB(self, A, B):
        dis = math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)
        return dis

    def cross(self, a, b):
        return a[0] * b[1] - b[0] * a[1]

    def calcAngle(self, a, b):  # Calculate angle between two vectors
        dot_product = a[0] * b[0] + a[1] * b[1]
        cos_alpha = dot_product / (sqrt(a[0] * a[0] + a[1] * a[1]) * sqrt(b[0] * b[0] + b[1] * b[1]))
        alpha = acos(cos_alpha) * 180 / pi
        if (self.cross(a, b - a) < 0): alpha *= -1
        return alpha

    def rotate(self, vector, fingerCount):
        if self.prevState != States.Rotate or fingerCount != 2:
            self.prevState = States.Rotate
            self.vector = vector
        else:
            height, width = self.image.shape[:2]
            deltaAlpha = self.calcAngle(self.vector, vector)
            rotation_matrix = cv.getRotationMatrix2D((width / 2, height / 2), deltaAlpha, 1)

            self.image = cv.warpAffine(self.image, rotation_matrix, (width, height))
            self.vector = vector
