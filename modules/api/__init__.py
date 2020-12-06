"""Module with methods for app creation"""


from flask import Flask, g

from modules.api.api_methods import blueprint


def create_app(model_config_dict) -> Flask:
    """
    Creates app

    :return: app class
    """

    app = Flask(__name__)

    app.config['model_config_dict'] = model_config_dict

    app.register_blueprint(blueprint)

    return app


