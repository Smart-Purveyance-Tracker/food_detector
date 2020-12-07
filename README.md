# Food detector

Project for food detection using YoloV3 network

## Getting Started

To download project:
```
git clone https://github.com/Smart-Purveyance-Tracker/food_detector.git
```

To download model weights and configs:

```
bash deploy/download_model_data.sh
```

## Installation

### Using local environment

1. Install requirements

    ```
    pip install -r requirements.txt
    ```

2. Set environment variables

    Example of vars:

    ```
    # Required params. You need to set them
   
    MODEL_PATH=/app/data/model_weights/best.pt  # path to the model
    MODEL_CONFIG_PATH=/app/data/model_configs/yolov3-spp2.cfg  # path to the model config
    
    CLASS_NAMES_PATH=/app/data/model_configs/names.txt  # path to the names of classes
    
    # Optional params. They will be set automatically to values provided below.
   
    IMAGE_SIZE=640
    CONF_THRESH=0.3
    IOU_THRESH=0.6
    AUGMENT=True
    
    PORT=9000
    HOST=0.0.0.0
    ```

3. Execute model:

    a) Start API
    
        python3 start_api.py

    b) Execute on an image locally. It will show image as the result.

        python3 test_model.py --image-path /home/vadbeg/Data/Docker_mounts/food/_108997168_766685d49e_o.jpg


### Using docker

1. Build docker image:

    ```
    bash build_docker.sh
    ```

2. Run docker container:

    ```
    # first param - port
    # second param - path to the volume with model weights, config, class_names
    bash run_docker.sh 7000 /home/user/Projects/Pet/food_detector/data/
    ```

### API example:

```python
import requests

url = 'http://0.0.0.0:9000/process_image'
my_image = {'image': open('/home/user/Data/Docker_mounts/food/_108997168_766685d49e_o.jpg', mode='rb')}

requests.post(url=url, files=my_image).json()
```

Result:

```python
[{'cls': 'Orange',
  'conf': 0.3254798352718353,
  'xyxy': [140.0, 105.0, 186.0, 165.0]},
 {'cls': 'Apple',
  'conf': 0.3627054989337921,
  'xyxy': [225.0, 123.0, 271.0, 173.0]},
 {'cls': 'Orange',
  'conf': 0.37995296716690063,
  'xyxy': [183.0, 101.0, 235.0, 166.0]}]
```


## Built With

* [PyTorch](https://flask.palletsprojects.com/en/1.1.x/) - Framework for neural nets.
* [numpy](https://flask.palletsprojects.com/en/1.1.x/) - The math framework used.

## Authors

* **Vadim Titko** aka *Vadbeg* - [GitHub](https://github.com/Vadbeg) 
| [LinkedIn](https://www.linkedin.com/in/vadtitko/)