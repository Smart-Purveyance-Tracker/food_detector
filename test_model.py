import argparse

import numpy as np
from cv2 import cv2

from modules.model.create_model import YoloModel
from modules.utils import parse_model_config, draw_image_with_bboxes


def read_image(image_path: str) -> np.ndarray:
    """
    Reads image from give path

    :param image_path: path to the image
    :return: cv2 image
    """

    image = cv2.imread(filename=image_path)

    return image


if __name__ == '__main__':
    """
    Example:
    
    >> python3 test_model.py \
    >>     --image-path /home/vadbeg/Data/Docker_mounts/food/_108997168_766685d49e_o.jpg \
    >>     --config-path config.ini
    
    """

    parser = argparse.ArgumentParser(description='Executes model on give image')

    parser.add_argument('--image-path', help='Path to the image', type=str)
    parser.add_argument('--config-path', help='Path to the config', type=str)

    args = parser.parse_args()

    image_path = args.image_path
    config_path = args.config_path

    model_config_dict = parse_model_config(model_config_path=config_path)

    model = YoloModel(
        model_path=model_config_dict['model_path'],
        model_config_path=model_config_dict['model_config_path'],
        image_size=model_config_dict['image_size'],
        conf_thresh=model_config_dict['conf_thresh'],
        iou_thresh=model_config_dict['iou_thresh'],
        augment=model_config_dict['augment']
    )

    image = read_image(image_path=image_path)

    res = model.predict(image=image)

    draw_image_with_bboxes(image=image, bboxes_list=res)
