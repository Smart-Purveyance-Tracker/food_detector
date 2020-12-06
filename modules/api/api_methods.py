"""Module with main api calls"""


import numpy as np
from cv2 import cv2
from PIL import Image
from flask import Blueprint, g, request, jsonify, current_app

from modules.model.create_model import YoloModel
from modules.api.model_access import get_model


blueprint = Blueprint('food_api', __name__)


@blueprint.route('/process_image', methods=['POST'])
def process_image():
    """Process image and reads"""
    file = request.files.get('image', '').read()

    image_buffer = np.fromstring(file, np.uint8)
    image = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)

    with current_app.app_context():
        model: YoloModel = get_model()

    prediction = model.predict(image=image)

    print(f'Prediction: {prediction}')

    return jsonify(prediction)

