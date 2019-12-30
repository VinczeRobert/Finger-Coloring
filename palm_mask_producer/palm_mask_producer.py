import numpy as np
import cv2


class PalmMaskProducer:

    def __init__(self, P, F):
        self._samples = []
        self._dx = [-1, 0, 1, -1, 1, -1, 0, 1]
        self._dy = [-1, -1, -1, 0, 0, 1, 1, 1]
        self._mask_points = []
        self._P = P
        self._F = F

    def compute_samples(self, X0, Y0, R, sampling_step):
        self._samples = []
        for index in range(0, 360, sampling_step):
            X = int(R * np.cos(index * np.pi / 180.0) + X0)
            Y = int(R * np.sin(index * np.pi / 180.0) + Y0)
            self._samples.append((Y, X))
            cv2.circle(self._F, (Y, X), 2, (255, 255, 0), -1)

        cv2.imshow('Image with Samples', self._F)

    def find_nearest_boundary_point(self, point):
        angle = 0
        rad = 1
        while angle <= 360 and rad <= self._P.shape[0]:
            x = int(np.cos(angle * np.pi / 180.0) * rad + point[0])
            y = int(np.sin(angle * np.pi / 180.0) * rad + point[1])

            if x >= 0 and y >= 0 and x < self._P.shape[0] and y < self._P.shape[0] and self._P[x][y] == 0:
                for k in range(8):
                    if 0 <= x + self._dx[k] < self._P.shape[0] and 0 <= y + self._dy[k] < self._P.shape[1] and \
                            self._P[x + self._dx[k]][y + self._dy[k]] == 0:
                        self._mask_points.append((x + self._dx[k], y + self._dy[k]))
            angle = angle + 1
            rad = rad + 1



    def get_palm_mask(self):
        self._mask_points = []
        for point in self._samples:
            self.find_nearest_boundary_point(point)

        return self._mask_points
