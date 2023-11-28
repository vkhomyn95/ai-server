#!/bin/bash
docker rm -f ai-server
docker build -t vk/ai-server .
docker run -d --restart always --net=host -v /stor:/stor -v /etc/localtime:/etc/localtime:ro \
        --log-opt max-size=500m --log-opt max-file=5 \
        -e APP_HOST='127.0.0.1' \
        -e APP_PORT='50053' \
        -e AUDIO_DIR='/stor/data/audio/' \
        -e LOGGER_DIR='/stor/data/logs/' \
        -e DEFAULT_MAX_AUDIO_INTERVAL='2.0' \
        -e DEFAULT_AUDIO_RATE=8000 \
        -e DEFAULT_AUDIO_SILENCE_EXCLUDE_INTERVAL=0.4 \
        -e DATABASE_HOST='127.0.0.1' \
        -e DATABASE_USER='root' \
        -e DATABASE_PASSWORD='root' \
        -e DATABASE_NAME='amd' \
        -e DATABASE_PORT=3306 \
        -e PYTHONUNBUFFERED=0 \
        --name ai-server vk/ai-server
