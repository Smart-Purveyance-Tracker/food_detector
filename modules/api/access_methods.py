"""Module with model access functions"""

from typing import Dict, List, Union

from flask import g, current_app
from modules.model.create_model import YoloModel
from modules.utils import class_names_mapping


def create_model(model_config_dict: Dict[str, Union[str, float, int, bool]]) -> YoloModel:
    """
    Creates YOLO model

    :param model_config_dict: config with model settings
    :return: model instance
    """

    model = YoloModel(
        model_path=model_config_dict['model_path'],
        model_config_path=model_config_dict['model_config_path'],
        image_size=model_config_dict['image_size'],
        conf_thresh=model_config_dict['conf_thresh'],
        iou_thresh=model_config_dict['iou_thresh'],
        augment=model_config_dict['augment']
    )

    return model


def create_class_names_list(model_config_dict: Dict[str, Union[str, float, int, bool]]) -> List[str]:
    """
    Creates YOLO model

    :param model_config_dict: config with model settings
    :return: model instance
    """
    class_names_path = model_config_dict['class_names_path']

    class_names_list = class_names_mapping(class_names_path=class_names_path)

    return class_names_list


def get_model():
    if 'model' not in g:
        g.model = create_model(model_config_dict=current_app.config.get('model_config_dict'))

    return g.model


def get_class_names():
    if 'class_names' not in g:
        g.class_names = create_class_names_list(model_config_dict=current_app.config.get('model_config_dict'))

    return g.class_names
