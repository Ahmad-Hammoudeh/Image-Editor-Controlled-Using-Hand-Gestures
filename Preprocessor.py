import cv2 as cv
import numpy as np


class Preprocessor:
    def __init__(self):
        pass

    def run(self, frame):
        frame = cv.flip(frame, 1)
        # frame = cv.medianBlur(frame, 5)
        # frame = cv.bilateralFilter(frame, 9, 75, 75)
        # frame = cv.GaussianBlur(frame, (5, 5), 50)

        # frame = np.float32(frame)
        # frame = frame / 255
        # rows, cols, dim = frame.shape
        # rh, rl, cutoff = 2.5, 0.5, 32
        #
        # imgYCrCb = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)
        # y, cr, cb = cv.split(imgYCrCb)
        # y_log = np.log(y + 0.01)
        # y_fft = np.fft.fft2(y_log)
        # y_fft_shift = np.fft.fftshift(y_fft)
        #
        # DX = cols / cutoff
        # G = np.ones((rows, cols))
        # for i in range(rows):
        #     for j in range(cols):
        #         G[i][j] = ((rh - rl) * (1 - np.exp(-((i - rows / 2) ** 2 + (j - cols / 2) ** 2) / (2 * DX ** 2)))) + rl
        #
        # result_filter = G * y_fft_shift
        #
        # result_interm = np.real(np.fft.ifft2(np.fft.ifftshift(result_filter)))
        #
        # result = np.exp(result_interm)

        # gamma
        # invGamma = 0.9
        # table = np.array([((i / 255.0) ** invGamma) * 255
        #                   for i in np.arange(0, 256)]).astype("uint8")
        #
        # frame = cv.LUT(frame, table)
        return frame
