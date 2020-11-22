"""Module with methods for app creation"""

from flask import Flask

from modules.api.api_methods import blueprint


def create_app() -> Flask:
    """
    Creates app

    :return: app class
    """

    app = Flask(__name__)

    app.register_blueprint(blueprint)

    return app


