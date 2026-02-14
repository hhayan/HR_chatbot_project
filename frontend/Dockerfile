# Frontend Dockerfile — Streamlit UI
FROM python:3.12-slim

WORKDIR /app

# 의존성 먼저 설치 (캐시 레이어 활용)
COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY frontend/ .

# Render는 PORT 환경변수를 주입함. 기본값 10000 (Render Free 기본 포트)
ENV PORT=10000
EXPOSE ${PORT}

CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false"]
