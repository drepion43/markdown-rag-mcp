FROM python:3.12.8-alpine

WORKDIR /app

# Install curl, git and uv
RUN apk add --no-cache curl git && \
    pip install --upgrade pip && \
    pip install uv

# Copy all files into the container
COPY . /app

# uv가 잘못된 python 버전을 찾지 않도록 .venv, .uv 초기화
RUN rm -rf .venv .uv

# Use uv to install the package
RUN uv sync

# 실행 명령어
CMD ["uv", "run", "src/server.py"]