# Python 3.13 slim 버전 사용
FROM python:3.13-slim

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    default-jdk \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리
WORKDIR /app

# requirements.txt 복사 및 설치
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 앱 소스 복사
COPY . .

# Flask 환경 변수
ENV FLASK_APP=code/app.py
ENV FLASK_RUN_HOST=0.0.0.0

# 앱 실행
CMD ["flask", "run"]
