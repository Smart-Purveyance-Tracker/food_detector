"""Module with model class"""

import onnx
import onnxruntime as onnx_rt

import numpy as np
from cv2 import cv2


class YoloModel:
    """Yolo model base on ONNX model"""

    def __init__(self, model_path: str):
        self.model = self.build_model(model_path=model_path)

    @staticmethod
    def __reshape_image__(image: np.ndarray, image_shape=(352, 608)) -> np.ndarray:
        if len(image.shape) != 3:
            raise ValueError(f'Bad image passed. Image needs to have (n, h, w) shape. Current shape: {image.shape}')
        elif image.shape[2] > 4:
            raise ValueError(f'Bad image passed. Image needs to have 4 channels in max. Current shape: {image.shape}')

        image = cv2.resize(image, dsize=image_shape)

        image = np.transpose(image, axes=[2, 0, 1])
        image = np.expand_dims(image, axis=0)

        return image

    @staticmethod
    def __check_model__(model_path):
        """
        Checks if model is not corrupted

        :param model_path: path to the model
        """

        model = onnx.load(model_path)
        onnx.checker.check_model(model)

    def build_model(self, model_path):
        """
        Builds model from onnx file path

        :param model_path: path to the model
        :return: connection to the model
        """

        self.__check_model__(model_path=model_path)

        session = onnx_rt.InferenceSession(
            path_or_bytes=model_path
        )

        return session

    def predict(self, image: np.ndarray):
        image = self.__reshape_image__(image=image)

        input_name = self.model.get_inputs()[0].name

        classes_label_name = self.model.get_outputs()[0].name
        boxes_label_name = self.model.get_outputs()[1].name

        predictions = self.model.run([classes_label_name, boxes_label_name],
                                     {input_name: image.astype(np.float32)})

        classes_ohe, boxes_coords = predictions

        return classes_ohe, boxes_coords


if __name__ == '__main__':
    model_path_abs = '/home/vadbeg/Source/yolov3_copy/yolov3/weights/best.onnx'
    yolo_model = YoloModel(model_path=model_path_abs)

    image = cv2.imread(filename='/home/vadbeg/Source/yolov3/images/03843g_01_dunja.jpg')

    classes_ohe, boxes_coords = yolo_model.predict(image=image)

    # print(classes_ohe)
    # print(boxes_coords)
    #
    # print(classes_ohe.shape)
    # print(boxes_coords.shape)
    #
    # print(classes_ohe[0])
    # print(boxes_coords[0])

    res = np.max(classes_ohe, axis=1)

    print(np.sum(res >= 0.2) / res.shape[0])