"""Module with model class"""

from typing import List, Tuple, Union, Dict

import torch
import numpy as np

from modules.model.yolo_torch_utils.models import Darknet
from modules.model.yolo_torch_utils.datasets import letterbox
from modules.model.yolo_torch_utils.utils import (non_max_suppression,
                                                  scale_coords)


class YoloModel:
    """Yolo model base on ONNX model"""

    def __init__(self, model_path: str, model_config_path: str,
                 image_size: int, conf_thresh: float = 0.3, iou_thresh: float = 0.6,
                 augment: bool = True):
        """
        Default model constructor.

        :param model_path: path to the model
        :param model_config_path: path to the config
        :param image_size: image of the on side if image. NEEDS TO BE DIVIDABLE BY 32!
        :param conf_thresh: confidence threshold for bbox
        :param iou_thresh: iou threshold for non maximum suppression
        :param augment: if True performs TTA (it is slower)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = self.__build_model__(model_path=model_path,
                                          model_config_path=model_config_path,
                                          image_size=image_size)

        self.image_size = image_size

        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.augment = augment

    def __reshape_image__(self, image: np.ndarray) -> np.ndarray:
        """
        Reshapes image for YOLO model

        :param image: original image
        :return: reshaped image
        """

        if len(image.shape) != 3:
            raise ValueError(f'Bad image passed. Image needs to have (n, h, w) shape. Current shape: {image.shape}')
        elif image.shape[2] > 4:
            raise ValueError(f'Bad image passed. Image needs to have 4 channels in max. Current shape: {image.shape}')

        image_shape = (self.image_size, self.image_size)

        image = letterbox(image, new_shape=image_shape)[0]

        image = image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        image = np.ascontiguousarray(image)

        image = torch.from_numpy(image).to(self.device)
        image = image.float()  # uint8 to fp16/32
        image /= 255.0  # 0 - 255 to 0.0 - 1.0
        if image.ndimension() == 3:
            image = image.unsqueeze(0)

        return image

    def __build_model__(self, model_path: str, model_config_path: str, image_size: int):
        """
        Builds model from onnx file path

        :param model_path: path to the model
        :param model_config_path: path to the model config
        :param image_size: size of the image
        :return: PyTorch model instance
        """

        model = Darknet(model_config_path, image_size)

        if model_path.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(model_path, map_location=self.device)['model'])
        else:
            raise ValueError(f'Bad model file extension: {model_path}. Needs to be *.pt')

        model.to(self.device).eval()

        return model

    def predict(self, image: np.ndarray) -> List[Dict[str, Union[List[float], float]]]:
        """
        Performs prediction on give image

        :param image: image for prediction
        :return: list of bboxes
        """

        original_image_size = image.shape

        image = self.__reshape_image__(image=image)

        bboxes_list = self.__detect__(
            image=image,
            model=self.model,
            original_image_size=original_image_size,
            conf_thresh=self.conf_thresh,
            iou_thresh=self.iou_thresh,
            augment=self.augment,
        )

        return bboxes_list

    @staticmethod
    def __detect__(image: np.ndarray, model, original_image_size: Tuple[int, int],
                   conf_thresh: float = 0.3,
                   iou_thresh: float = 0.6, augment: bool = True) -> List[Dict[str, Union[List[float], float]]]:
        """
        Executes model on given image

        This function is highly rebuilded function from yolov3-ultralytics repo with yolov3.
        It uses function from yolo_torch_utils
        :source: https://github.com/ultralytics/yolov3

        :param image: normalized and reshaped image for model
        :param model: torch.nn.Module network
        :param original_image_size: size of the image side (image will be rectangular) for reshape
        :param conf_thresh: confidence threshold for bbox
        :param iou_thresh: iou threshold for non maximum suppression
        :param augment: if True performs TTA (it is slower)
        :return: list with dictionaries of bboxes
        """

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
                det[:, :4] = scale_coords(image.shape[2:], det[:, :4], original_image_size).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xyxy = [float(curr_coord.cpu().detach().numpy()) for curr_coord in xyxy]
                    conf = float(conf.cpu().detach().numpy())
                    cls = float(cls.cpu().detach().numpy())

                    bbox_dict = {
                        'xyxy': xyxy,
                        'conf': conf,
                        'cls': cls
                    }

                    image_bboxes_list.append(bbox_dict)

        return image_bboxes_list
