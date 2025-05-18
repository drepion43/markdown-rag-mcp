FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy source code
COPY pyproject.toml /app/

RUN python -c "import tomllib; deps = tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']; print('\n'.join(deps))" > requirements.txt
# Install dependencies using pip from the requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# 컨테이너 실행 시 사용할 명령어 지정
CMD ["python3", "src/server.py"]