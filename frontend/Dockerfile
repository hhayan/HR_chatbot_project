# Frontend Dockerfile — Streamlit UI
FROM python:3.12-slim

WORKDIR /app

# 루트에서 빌드하므로 frontend/requirements.txt를 복사
COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 루트에서 빌드하므로 frontend/ 폴더 전체를 복사
COPY frontend/ .

EXPOSE 8501

CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]
