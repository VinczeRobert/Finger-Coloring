import cv2
import numpy as np

class PalmPointCalculator():

    def __init__(self):
        self._max_i = -1
        self._max_j = -1
        self._max = 0

    def obtaining_palm_point(self, dt):

        for i in range(dt.shape[0]):
            for j in range(dt.shape[1]):
                if dt[i][j] > self._max:
                    self._max = dt[i][j]
                    self._max_i = i
                    self._max_j = j

        return self._max_i, self._max_j

    def reset(self):
        self._max_i = -1
        self._max_j = -1
        self._max = 0

    @staticmethod
    def convert_from_one_channel_to_three(image):
        row = []
        three_channeled = []

        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                row.append([image[i][j], image[i][j], image[i][j]])
            three_channeled.append(row)
            row = []

        return three_channeled

    def draw_image_with_palm_point(self, image):
        image_with_circle = np.uint8(image)
        cv2.circle(image_with_circle, (self._max_i, self._max_j), 5, (0, 255, 0), -1)
        cv2.imshow('Picture with Palm Point', image_with_circle)