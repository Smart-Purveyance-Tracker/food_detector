FROM python:3.8.6-buster

COPY . /app
WORKDIR /app

RUN apt update && apt -y install libgl1-mesa-glx

RUN pip3 install -r requirements.txt

ENTRYPOINT bash deploy/run_api.sh


