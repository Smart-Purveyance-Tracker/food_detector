"""Module with API starting"""

import argparse

from modules.api import create_app
from modules.utils import get_model_config, get_api_host_port


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Executes model on give image')

    model_config_dict = get_model_config()
    host, port = get_api_host_port()

    app = create_app(model_config_dict=model_config_dict)

    app.run(host=host, port=port)
