import cv2
import numpy as np

class PalmSegmentor():

    def __init__(self):
        self._max_i = -1
        self._max_j = -1
        self._maximum_radius = 0

    def obtaining_palm_point(self, dt):
        self._max_i, self._max_j = np.unravel_index(np.argmax(dt, axis=None), dt.shape)
        self._maximum_radius = dt[self._max_i][self._max_j]
        return self._max_i, self._max_j

    @staticmethod
    def from_one_channel_to_three(original_image, binary_image):
        three_channels = np.zeros_like(3 * original_image)
        three_channels[:,:,0] = binary_image
        three_channels[:,:,1] = binary_image
        three_channels[:,:,2] = binary_image

        return three_channels

    def draw_image_with_palm_point(self, image):
        cv2.circle(image, (self._max_j, self._max_i), 5, (0, 255, 0), -1)
        # cv2.imshow('Picture with Palm Point', image)

    def draw_image_with_inner_circle(self, image):
        # cv2.circle(image, (self._max_j, self._max_i), self._maximum_radius, (255, 0, 0), 2)
        # cv2.circle(image, (self._max_j, self._max_i), int(1.2*self._maximum_radius), (0, 0, 255), 2)
        cv2.rectangle(image, (self._max_j - int(1.2*self._maximum_radius), self._max_i - int(1.2*self._maximum_radius)),
                      (self._max_j + int(1.2*self._maximum_radius), self._max_i + int(1.2*self._maximum_radius)), (0, 0, 0), -1)
        cv2.imshow("Gigi", image)

    def get_maximum_radius_12(self):
        return int(1.2 * self._maximum_radius)