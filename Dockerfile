FROM python:3.10

WORKDIR /usr/src

COPY requirements.txt requirements.txt

RUN python -m venv venv

RUN python -m pip install grpcio

RUN python -m pip install grpcio-tools

RUN python -m pip install protobuf

RUN python -m pip install googleapis-common-protos

RUN python -m pip install PyJWT

RUN python -m pip install mariadb

RUN python -m pip install librosa

RUN python -m pip install matplotlib

RUN python -m pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

RUN python -m pip install "numpy<2.0"

RUN venv/bin/pip3 install --no-cache-dir -r requirements.txt

COPY . ./

EXPOSE 50053

ENTRYPOINT ["python3", "-u", "stt_server.py"]
