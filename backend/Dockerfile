# Backend Dockerfile — FastAPI + LangGraph
FROM python:3.12-slim

# Root Context 기준: WORKDIR을 /app/backend로 설정해도 되지만, 
# 관례상 컨테이너 내부는 /app을 많이 사용.
# 사용자 요청대로 소스 구조에 맞게 복사.

WORKDIR /app

# 루트에서 빌드하므로 backend/requirements.txt를 복사
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 루트에서 빌드하므로 backend/ 폴더 전체를 복사
COPY backend/ .

EXPOSE 8000

# ENV PORT=8000 (Optional: Dockerfile 내 기본값 설정을 명시할 수도 있음)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
