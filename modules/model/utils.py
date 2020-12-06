"""
Module with utils for model execution

PLS don't try to change it!

:source: https://github.com/ultralytics/yolov3
"""

from typing import List, Union, Dict

import torch
import numpy as np

from modules.model.yolo_torch_utils.models import Darknet
from modules.model.yolo_torch_utils.datasets import letterbox
from modules.model.yolo_torch_utils.utils import (non_max_suppression,
                                                  scale_coords,
                                                  load_classes)


def detect(image: np.ndarray, weights_path: str = '', image_size: int = 640,
           model_config_path: str = '', class_names_path: str = '', conf_thresh: float = 0.3,
           iou_thresh: float = 0.6, augment: bool = True) -> List[Dict[str, Union[List[float], float]]]:
    """
    Executes model on given image

    :param image: image for model
    :param weights_path: path to the weights
    :param image_size: size of the image side (image will be rectangular) for reshape
    :param model_config_path: path to the model config
    :param class_names_path: path to txt file with class names
    :param conf_thresh: confidence threshold for bbox
    :param iou_thresh: ioy threshold for non maximum suppression
    :param augment: if True performs TTA (it is slower)
    :return: list with dictionaries of bboxes
    """

    original_shape = image.shape

    # Initialize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = Darknet(model_config_path, image_size)

    # Load weights
    if weights_path.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights_path, map_location=device)['model'])
    else:
        raise Exception

    # Eval mode
    model.to(device).eval()

    # Get names and colors
    names = load_classes(class_names_path)

    image = letterbox(image, new_shape=(image_size, image_size))[0]

    # Convert image
    image = image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    image = np.ascontiguousarray(image)

    image = torch.from_numpy(image).to(device)
    image = image.float()  # uint8 to fp16/32
    image /= 255.0  # 0 - 255 to 0.0 - 1.0
    if image.ndimension() == 3:
        image = image.unsqueeze(0)

    # Inference
    pred = model(image, augment=augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thresh, iou_thresh,
                               multi_label=False, classes=None, agnostic=True)

    image_bboxes_list = list()

    # Process detections
    for i, det in enumerate(pred):  # detections for image i

        if det is not None and len(det):
            # Rescale boxes from imgsz to im0 size
            det[:, :4] = scale_coords(image.shape[2:], det[:, :4], original_shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                xyxy = [curr_coord.cpu().detach().numpy() for curr_coord in xyxy]
                conf = conf.cpu().detach().numpy()
                cls = cls.cpu().detach().numpy()

                bbox_dict = {
                    'xyxy': xyxy,
                    'conf': conf,
                    'cls': cls
                }

                image_bboxes_list.append(bbox_dict)

    return image_bboxes_list
