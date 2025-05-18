# Python 3.12.4-slim을 기반으로 사용 (Ubuntu 대신 경량 이미지)
FROM python:3.12.4-slim

WORKDIR /app

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1 \
    OPENAI_API_KEY="" \
    PATH="/root/.cargo/bin:$PATH"

# uv 설치
RUN apt-get update && \
    apt-get install -y curl && \
    curl -LsSf https://astral.sh/uv/install.sh -o install_uv.sh && \
    chmod +x install_uv.sh && \
    ./install_uv.sh && \
    rm install_uv.sh && \
    export PATH=/root/.local/bin:$PATH && \
    uv --version && \
    apt-get remove -y curl && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# 의존성 파일만 먼저 복사 (Docker 캐싱 활용)
COPY pyproject.toml uv.lock /app/

# 의존성 설치
RUN export PATH=/root/.local/bin:$PATH && \
    uv sync --frozen --no-cache

# 나머지 소스 코드 복사
COPY . /app

# 실행 명령어
CMD ["uv", "run", "python3", "src/server.py"]