FROM python:3.12.8-slim

WORKDIR /app

# Install curl, git and uv
RUN pip install --upgrade pip && \
    pip install uv

# Copy all files into the container
COPY . /app

# Use uv to install the package
RUN uv sync

# 실행 명령어
CMD ["uv", "run", "src/server.py"]