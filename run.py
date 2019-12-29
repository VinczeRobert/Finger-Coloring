import cv2
import numpy as np

from background_subtraction.background_subtraction import BackgroundSubtractor
from constants import get_constants
from conversion_to_binary.conversion_to_binary import BinaryConvertor
from distance_transform.distance_transform import DistanceTransformCalculator
from frame_obtaining.frame_obtaining import FrameObtainer
from palm_mask_producer.palm_mask_producer import PalmMaskProducer
from palm_segmentor.palm_segmentor import PalmSegmentor


if __name__ == '__main__':
    params = get_constants()

    frame_obtainer = FrameObtainer(params['cap_region_x_begin'],
                                   params['cap_region_y_end'],
                                   params['threshold'],
                                   params['blur_value'],
                                   params['background_sub_threshold'],
                                   params['eta'])
    frame_obtainer.create_trackbar()
    background_subtractor = BackgroundSubtractor(params['background_sub_threshold'],
                                                 params['eta'])

    distrance_transform_calculator = DistanceTransformCalculator()
    palm_point_segmentor = PalmSegmentor()

    while frame_obtainer.get_camera().isOpened():
        original_image = frame_obtainer.read_frame()

        if background_subtractor.is_background_captured():
            background_subtractor.set_frame(original_image)
            background = background_subtractor.extract_background(params['cap_region_y_end'],
                                                                  params['cap_region_x_begin'])
            binary_convertor = BinaryConvertor(background)
            binary_image = binary_convertor.convert_to_binary(params['blur_value'], params['threshold'])
            binary_image = cv2.resize(binary_image, (200, 200), interpolation=cv2.INTER_AREA)
            dt = distrance_transform_calculator.calculate_distance_transform(binary_image)
            max_i, max_j = palm_point_segmentor.obtaining_palm_point(dt)
            background = cv2.resize(background, (200, 200), interpolation=cv2.INTER_AREA)
            image_with_palm_point = palm_point_segmentor.from_one_channel_to_three(background, binary_image)
            palm_point_segmentor.draw_image_with_palm_point(image_with_palm_point)
            palm_point_segmentor.draw_image_with_inner_circle(image_with_palm_point)

            # palm_mask_producer = PalmMaskProducer(binary_image, image_with_palm_point)
            # palm_mask_producer.compute_samples(max_i, max_j, palm_point_segmentor.get_maximum_radius_12(), params['sampling_step'])
            # mask_points = palm_mask_producer.get_palm_mask()

        k = cv2.waitKey(10)
        if k == 27:
            frame_obtainer.get_camera().release()
            cv2.destroyAllWindows()
            break
        # press B to capture background
        elif k == ord('b'):
            background_subtractor.set_background_captured(True)
            print('! Background Captured!')
        elif k == ord('r'):
            background_subtractor.set_background_captured(False)
            print('! Background Reset!')
