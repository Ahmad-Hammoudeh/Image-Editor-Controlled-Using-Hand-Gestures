import cv2 as cv
import numpy as np
from itertools import combinations_with_replacement
from collections import defaultdict
from numpy.linalg import inv
from PIL import Image, ImageEnhance
from PIL import Image as im


class Preprocessor:
    def __init__(self):
        pass

    def run(self, frame):
        frame = cv.flip(frame, 1)
        # frame = cv2.medianBlur(frame, 5)
        # frame = cv.bilateralFilter(frame, 9, 75, 75)
        # frame = cv2.GaussianBlur(frame, (5, 5), 75)

        # image brightness enhancer
        # data = im.fromarray(frame)
        #
        # enhancer = ImageEnhance.Brightness(data)
        # factor = 1.2  # brightens the image
        # im_output = enhancer.enhance(factor)
        # result = np.asarray(im_output)
        #
        # result = cv.medianBlur(result, 5)
        # result = cv.bilateralFilter(result, 9, 75, 75)
        # result = cv.GaussianBlur(result, (5, 5), 75)
        # cv.imshow("en", result)

        # factor = 1  # gives original image
        # im_output = enhancer.enhance(factor)
        # im_output.save('original-image.png')
        #
        # factor = 0.5  # darkens the image
        # im_output = enhancer.enhance(factor)
        # im_output.save('darkened-image.png')

        return frame
