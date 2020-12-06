"""Module with utils for whole project"""

import configparser
from typing import Dict, Union, List

import numpy as np
from cv2 import cv2


def parse_model_config(config_path: str) -> Dict[str, Union[str, float, int, bool]]:
    config = configparser.ConfigParser()
    config.read(filenames=config_path)

    model_path = config['Model']['model_path']
    model_config_path = config['Model']['model_config_path']

    image_size = config.getfloat('Model', 'image_size')
    conf_thresh = config.getfloat('Model', 'conf_thresh')
    iou_thresh = config.getfloat('Model', 'iou_thresh')
    augment = config.getboolean('Model', 'augment')

    result = {
        'model_path': model_path,
        'model_config_path': model_config_path,

        'image_size': image_size,
        'conf_thresh': conf_thresh,
        'iou_thresh': iou_thresh,
        'augment': augment
    }

    return result


def draw_image_with_bboxes(image: np.ndarray, bboxes_list: List[Dict[str, Union[List[float], float]]]):
    """
    Draw image with bboxes and shows it.

    :param image: image to draw
    :param bboxes_list: list of bboxes to draw
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

    cv2.imshow('Image', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


