import cv2

from background_subtraction.background_subtraction import BackgroundSubtractor
from constants import get_constants
from conversion_to_binary.conversion_to_binary import BinaryConvertor
from distance_transform.distance_transform import DistanceTransformCalculator
from frame_obtaining.frame_obtaining import FrameObtainer
from palm_point.palm_point import PalmPointCalculator


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
    palm_point_calculator = PalmPointCalculator()

    while frame_obtainer.get_camera().isOpened():
        original_image = frame_obtainer.read_frame()

        if background_subtractor.is_background_captured():
            background_subtractor.set_frame(original_image)
            background = background_subtractor.extract_background(params['cap_region_y_end'],
                                                                  params['cap_region_x_begin'])
            binary_convertor = BinaryConvertor(background)
            binary_image = binary_convertor.convert_to_binary(params['blur_value'],
                                                              params['threshold'])

            dt = distrance_transform_calculator.calculate_distance_transform(binary_image)
            max_i, max_j = palm_point_calculator.obtaining_palm_point(dt)
            palm_point_calculator.draw_image_with_palm_point(background)

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
