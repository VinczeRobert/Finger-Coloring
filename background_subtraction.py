import cv2
import numpy as np


def print_treshold(thr):
    print("! Changed threshold to " + str(thr))

def obtaining_palm_point(input):
    max = 0
    max_i = -1
    max_j = -1

    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            if input[i][j] > max:
                max = input[i][j]
                max_i = i
                max_j = j

    return max_i, max_j

# needed for obtaining the palm point
def distance_transform(input, type, mask_size):
    h, w = input.shape[:2]
    dst = np.zeros((h, w), np.float32)
    cv2.distanceTransform(input, type, mask_size, dst=dst)
    return np.uint8(dst)

def extract_background(frame, bgModel, eta):
    # get mask
    mask = bgModel.apply(frame, learningRate=eta)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    # get background by checking difference
    res = cv2.bitwise_and(frame, frame, mask=mask)
    return res


# this function is only called if the background was already subtracted
def get_hand(frame, cap_region_y_end, cap_region_x_begin, blur_value, threshold, bgModel, eta):
    img = extract_background(frame, bgModel, eta)
    # extract the region of interest
    img = img[0:int(cap_region_y_end * frame.shape[0]),
          int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]
    cv2.imshow('Extracted Hand', frame)

    # converting the image to binary
    # hand should be white, rest should be black
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_value, blur_value), 0)
    _, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
    cv2.imshow('Binary Image', thresh)
    return thresh

def from_one_channel_to_three(one_channeled):
    row = []
    three_channel = []
    for i in range(0, one_channeled.shape[0]):
        for j in range(0, one_channeled.shape[1]):
            row.append([one_channeled[i][j], one_channeled[i][j], one_channeled[i][j]])
        three_channel.append(row)
        row = []

    return three_channel

def run():
    # alter parameters
    cap_region_x_begin = 0.5
    cap_region_y_end = 0.5
    threshold = 60
    blur_value = 17
    bg_sub_threshold = 50
    eta = 0

    is_bg_captured = False
    trackbar_name = 'Finger Coloring Application'

    # Camera
    camera = cv2.VideoCapture(0)
    camera.set(10, 200)
    cv2.namedWindow(trackbar_name)
    cv2.createTrackbar('thr', trackbar_name, threshold, 100, print_treshold)

    while camera.isOpened():
        ret, frame = camera.read()
        threshold = cv2.getTrackbarPos('thr', trackbar_name)
        # use smoothing filter
        frame = cv2.bilateralFilter(frame, 5, 50, 100)
        frame = cv2.flip(frame, 1)
        # flip the frame horizontally
        cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                      (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (0, 255, 0), 2)
        cv2.imshow('Original Image', frame)

        if is_bg_captured:
            image = get_hand(frame, cap_region_y_end, cap_region_x_begin, blur_value, threshold, bgModel, eta)
            resized_image = cv2.resize(image, dsize=(200, 200), interpolation=cv2.INTER_CUBIC)
            dt = np.array(distance_transform(resized_image, cv2.DIST_L2, 5))
            #max_i and max_j hold the coordinates of the palm point
            max_i, max_j = obtaining_palm_point(dt)
            image_with_circle = np.uint8(from_one_channel_to_three(resized_image))
            cv2.circle(np.uint8(image_with_circle), (max_i, max_j), 5, (0, 255, 0), -1)
            cv2.imshow('Picture with Palm point', image_with_circle)

        k = cv2.waitKey(10)
        # press ESC to exit programim
        if k == 27:
            camera.release()
            cv2.destroyAllWindows()
            break
        # press B to capture background
        elif k == ord('b'):
            bgModel = cv2.createBackgroundSubtractorMOG2(0, bg_sub_threshold)
            is_bg_captured = True
            print('! Background Captured!')
        elif k == ord('r'):
            bgModel = None
            is_bg_captured = False
            print('! Background Reset!')


if __name__ == '__main__':
    run()