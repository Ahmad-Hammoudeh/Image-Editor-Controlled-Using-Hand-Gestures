import cv2
import numpy as np
from itertools import combinations_with_replacement
from collections import defaultdict
from numpy.linalg import inv

class Preprocessor:
    def __init__(self):
        pass

    def run(self, frame):
        frame = cv2.flip(frame, 1)

        # frame = cv.medianBlur(frame, 5)
        # frame = cv.bilateralFilter(frame, 9, 75, 75)
        # frame = cv.GaussianBlur(frame, (5, 5), 50)

        return frame
