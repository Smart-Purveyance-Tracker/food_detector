"""Module with API starting"""

import argparse

from modules.api import create_app
from modules.utils import parse_model_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Executes model on give image')

    parser.add_argument('--host', help='API host', type=str, default='0.0.0.0')
    parser.add_argument('--port', help='Port for APE', type=str, default=9000)
    parser.add_argument('--config-path', help='Path to the config', type=str)

    args = parser.parse_args()

    host = args.host
    port = args.port
    config_path = args.config_path

    model_config_dict = parse_model_config(config_path=config_path)

    app = create_app(model_config_dict=model_config_dict)

    app.run(host=host, port=port)
