FROM python:3.10

WORKDIR /usr/src

COPY requirements.txt requirements.txt

RUN python -m venv venv

RUN venv/bin/pip3 install -r requirements.txt

COPY . ./

EXPOSE 50053
ENTRYPOINT ["python3", "-u", "stt_server.py"]
