"""Module with model access functions"""

from typing import Dict, Union

from flask import g, current_app
from modules.model.create_model import YoloModel


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


def get_model():
    if 'model' not in g:
        g.model = create_model(model_config_dict=current_app.config.get('model_config_dict'))

    return g.model
