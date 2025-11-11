# 1️⃣ 베이스 이미지: Python 3.13 slim
FROM python:3.13-slim

# 2️⃣ 시스템 의존성 설치 (Java 포함)
RUN apt-get update && apt-get install -y \
    openjdk-17-jre-headless \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 3️⃣ JAVA_HOME 환경 변수 설정
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"
ENV _JAVA_OPTIONS="-Xms64m -Xmx256m"

# 4️⃣ 작업 디렉토리
WORKDIR /app

# 5️⃣ requirements.txt 복사 및 설치
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 6️⃣ 앱 소스 복사
COPY . .

# 7️⃣ Flask 환경 변수
ENV FLASK_APP=code/app.py
ENV FLASK_RUN_HOST=0.0.0.0

# 8️⃣ 앱 실행
CMD ["flask", "run"]
