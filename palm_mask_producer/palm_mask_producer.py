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
            self._samples.append((Y,X))
            cv2.circle(self._F, (Y,X), 2, (255, 255, 0), -1)

        cv2.imshow('Image with Samples', self._F)

    def find_nearest_boundary_point(self, point):
        angle = 0
        rad = 1
        while angle < 360 and rad < self._P.shape[0]:
            while True:
                x = int(np.cos(angle * np.pi / 180.0) * rad + point[0])
                y = int(np.sin(angle * np.pi / 180.0) * rad + point[1])

                if x<0 or x>=200 or y<0 or y>=200:
                    break

                if(self._P[x][y] == 0):
                    break
                else:
                    rad = rad + 1
                    angle = angle + 1

            for i in range(8):
                if(x + self._dx[i] >= 0 and x + self._dx[i] < self._P.shape[0] and
                   y + self._dy[i] >= 0 and y + self._dy[i] < self._P.shape[0]):
                    if self._P[x + self._dx[i]][y + self._dy[i]] == 255:
                        self._mask_points.append((x + self._dx[i], y + self._dy[i]))
                        cv2.line(self._F, (point[0], point[1]), (x+self._dx[i], y+self._dy[i]), (255, 255, 0))
                        break
            angle = angle + 1
            rad = rad + 1

    def get_palm_mask(self):
        self._mask_points = []
        for point in self._samples:
            self.find_nearest_boundary_point(point)
        cv2.imshow('Palm Mask', self._F)
        return self._mask_points







