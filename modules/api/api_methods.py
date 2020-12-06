"""Module with main api calls"""


import numpy as np
from cv2 import cv2
from flask import Blueprint, request, jsonify, current_app

from modules.model.create_model import YoloModel
from modules.utils import change_idx_to_class_names
from modules.api.access_methods import get_model, get_class_names


blueprint = Blueprint('food_api', __name__)


@blueprint.route('/process_image', methods=['POST'])
def process_image():
    """Process image and reads"""
    file = request.files.get('image', '').read()

    image_buffer = np.fromstring(file, np.uint8)
    image = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)

    with current_app.app_context():
        model: YoloModel = get_model()
        class_names_list = get_class_names()

    prediction = model.predict(image=image)
    prediction = change_idx_to_class_names(bboxes_list=prediction, class_names=class_names_list)

    return jsonify(prediction)

