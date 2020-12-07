"""Module with utils for whole project"""

from typing import Dict, Union, List, Tuple

import environ
import numpy as np
from cv2 import cv2


def get_model_config() -> Dict[str, Union[str, float, int, bool]]:
    """
    Reads model config from environmental vas

    :return: dictionary with config
    """

    env = environ.Env()

    model_path = env.str('MODEL_PATH')
    model_config_path = env.str('MODEL_CONFIG_PATH')
    class_names_path = env.str('CLASS_NAMES_PATH')

    image_size = env.int('IMAGE_SIZE', default=640)
    conf_thresh = env.float('CONF_THRESH', default=0.3)
    iou_thresh = env.float('IOU_THRESH', default=0.6)
    augment = env.bool('AUGMENT', default=True)

    result = {
        'model_path': model_path,
        'model_config_path': model_config_path,
        'class_names_path': class_names_path,

        'image_size': image_size,
        'conf_thresh': conf_thresh,
        'iou_thresh': iou_thresh,
        'augment': augment
    }

    return result


def get_api_host_port() -> Tuple[str, int]:
    """
    Reads HOST, PORT environmental vas

    :return: host, port
    """

    env = environ.Env()

    host = env.str('HOST', default='0.0.0.0')
    port = env.int('PORT', default=9000)

    return host, port


def draw_image_with_bboxes(image: np.ndarray,
                           bboxes_list: List[Dict[str, Union[List[float], float]]],
                           upsale: bool = False):
    """
    Draw image with bboxes and shows it.

    :param image: image to draw
    :param bboxes_list: list of bboxes to draw
    :param upsale: if True makes image 2 time bigger
    """

    for curr_bbox_dict in bboxes_list:
        xyxy = curr_bbox_dict['xyxy']
        conf = curr_bbox_dict['conf']
        cls = curr_bbox_dict['cls']

        xyxy = map(int, xyxy)
        x1, y1, x2, y2 = xyxy

        text = f'{cls} {round(conf, 2)}'

        image = cv2.rectangle(image, (x1, y1), (x2, y2), (36, 255, 12), 1)
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.4, (36, 255, 12), 1)

    if upsale:
        new_size = (image.shape[1] * 2, image.shape[0] * 2)
        image = cv2.resize(image, new_size)

    cv2.imshow('Image', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def read_class_names_mapping(class_names_path: str) -> List[str]:
    """
    Returns list with class names (indexes are mapped to names because of ordering)

    :param class_names_path: path to class names
    :return: list of class names
    """

    with open(class_names_path, mode='r') as file:
        lines = file.readlines()
        lines = list(map(lambda x: x.strip(), lines))

    return lines


def change_idx_to_class_names(bboxes_list: List[Dict[str, Union[List[float], float]]],
                              class_names: List[str]) -> List[Dict[str, Union[List[float], float, str]]]:
    """
    Changes prediction integer labels to text labels

    :param bboxes_list: list of bboxes to draw
    :param class_names: list with class names
    :return: list of bboxes to draw with text label
    """

    for curr_bbox_dict in bboxes_list:
        cls = int(curr_bbox_dict['cls'])

        curr_bbox_dict['cls'] = class_names[cls]

    return bboxes_list


