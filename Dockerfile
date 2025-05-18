FROM ubuntu:24.04

#아래 쉘스크립트 에러 방지 - .bashrc 관련에서 에러
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# Set environment variables for configuration
ENV PYTHON_VERSION=3.12

# 기본 패키지 설치
RUN apt-get update -y && \
apt install software-properties-common -y && \
add-apt-repository ppa:deadsnakes/ppa -y && \
apt install python${PYTHON_VERSION} -y && \
apt-get install -y python3-pip && \
apt-get install nodejs npm -y && \
apt install -y curl&& \
apt-get clean

# uv install
RUN curl -LsSf https://astral.sh/uv/install.sh -o install_uv.sh && \
    chmod +x install_uv.sh && \
    ./install_uv.sh && \
    rm install_uv.sh

# 작업 디렉토리 설정 (변경 요소)
WORKDIR /app

COPY . /app

# 컨테이너 실행 시 사용할 명령어 지정
CMD ["uv run src/server.py"]