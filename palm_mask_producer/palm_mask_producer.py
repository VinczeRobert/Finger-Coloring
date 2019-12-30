import numpy as np
import cv2

dx = [-1, 0, 1, -1, 1, -1, 0, 1]
dy = [-1, -1, -1, 0, 0, 1, 1, 1]


def compute_samples(image, X0, Y0, R, sampling_step):
    samples = []
    for index in range(0, 360, sampling_step):
        X = int(R * np.cos(index * np.pi / 180.0) + X0)
        Y = int(R * np.sin(index * np.pi / 180.0) + Y0)
        samples.append((Y, X))
        cv2.circle(image, (Y, X), 2, (255, 255, 0), -1)

    cv2.imshow('Image with Samples', image)
    return samples

def get_palm_mask(P, samples):
    mask_points = []
    for point in samples:
        angle = 0
        rad = 1
        while angle <= 360 and rad <= P.shape[0]:
            x = int(np.cos(angle * np.pi / 180.0) * rad + point[0])
            y = int(np.sin(angle * np.pi / 180.0) * rad + point[1])

            if x >= 0 and y >= 0 and x < P.shape[0] and y < P.shape[0] and P[x][y] == 0:
                for k in range(8):
                    if 0 <= x + dx[k] < P.shape[0] and 0 <= y + dy[k] < P.shape[1] and \
                            P[x + dx[k]][y + dy[k]] == 0:
                        mask_points.append((x + dx[k], y + dy[k]))
            angle = angle + 1
            rad = rad + 1

    return mask_points
