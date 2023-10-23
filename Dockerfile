FROM python:3.10

RUN mkdir -p /usr/src/crmb
WORKDIR /usr/src/crmb

COPY requirements.txt /usr/src/crmb

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 50053
ENTRYPOINT ["python3", "-u", "stt_server.py"]
