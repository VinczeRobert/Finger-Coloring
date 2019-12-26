import numpy as np
import cv2

class PalmMaskProducer:

    def __init__(self):
        self.samples = []

    def compute_samples(self, X0, Y0, R, sampling_step, image):
        self.samples = []
        for index in range(0, 360, sampling_step):
            X = int(R * np.cos(index * np.pi / 180.0) + X0)
            Y = int(R * np.sin(index * np.pi / 180.0) + Y0)
            self.samples.append((X,Y))
            cv2.circle(image, (Y,X), 2, (255, 255, 0), -1)

        cv2.imshow('Image with Samples', image)

