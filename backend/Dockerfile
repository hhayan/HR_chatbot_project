# Backend Dockerfile — FastAPI + LangGraph
FROM python:3.12-slim

WORKDIR /app

# 의존성 먼저 설치 (캐시 레이어 활용)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY backend/ .

# Render는 PORT 환경변수를 주입함. 기본값 10000 (Render Free 기본 포트)
ENV PORT=10000
EXPOSE ${PORT}

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
