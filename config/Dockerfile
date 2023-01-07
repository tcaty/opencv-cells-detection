FROM continuumio/miniconda3:latest

WORKDIR /usr/src/miniconda_jupyter_server

COPY ../ .

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt
