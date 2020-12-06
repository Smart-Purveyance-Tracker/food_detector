FROM python:3.8.6-buster

ARG CONFIG_PATH=config.ini
ARG PORT=9000
ARG HOST='0.0.0.0'

ENV CONFIG_PATH ${CONFIG_PATH}
ENV PORT ${PORT}
ENV HOST ${HOST}

COPY . /app
WORKDIR /app

RUN apt update && apt -y install libgl1-mesa-glx

RUN pip3 install -r requirements.txt

ENTRYPOINT bash deploy/run_api.sh


