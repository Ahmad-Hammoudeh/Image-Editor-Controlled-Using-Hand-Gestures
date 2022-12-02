import cv2 as cv


class Preprocessor:
    def __init__(self):
        pass

    def run(self, frame):
        frame = cv.flip(frame, 1)
        #frame = cv.medianBlur(frame, 5)
        # frame = cv.bilateralFilter(frame, 9, 75, 75)
        frame = cv.GaussianBlur(frame, (5, 5), 100)
        return frame
