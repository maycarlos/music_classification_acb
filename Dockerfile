FROM python:3.9.16-slim-bullseye

RUN apt-get update && apt-get -y upgrade 

RUN useradd --create-home me
WORKDIR /home/me

USER me

COPY  music_classification /home/me/music_classification
COPY requirements.txt /home/me/
COPY setup.py /home/me/

RUN pip install -r requirements.txt

ENTRYPOINT [ "bash" ]
