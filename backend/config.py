# backend/config.py
"""
중앙 환경 변수 관리 (pydantic-settings)
- 프로젝트의 유일한 .env 로딩 지점
- 모든 모듈은 이 파일에서 settings를 import하여 사용
"""

from pathlib import Path
from pydantic_settings import BaseSettings

# 프로젝트 루트의 .env 경로 (backend/ 에서 실행해도 동작)
ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"


class Settings(BaseSettings):
    """환경 변수 기반 설정"""
    OPENAI_API_KEY: str = ""
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX: str = "gaida-hr-rules"

    # LLM 모델 설정
    GEN_LLM: str = "gpt-4.1"
    ROUTER1_LLM: str = "gpt-4.1-mini"
    ROUTER2_LLM: str = "gpt-4.1-nano"

    # CORS 허용 오리진 (쉼표 구분, 예: "https://foo.onrender.com,http://localhost:8501")
    ALLOWED_ORIGINS: str = "*"

    class Config:
        # 로컬: .env 파일에서 로드 / Docker: env_file 주입이므로 파일 불필요
        env_file = str(ENV_PATH) if ENV_PATH.exists() else None
        extra = "ignore"  # LANGSMITH_* 등 미정의 변수 무시


settings = Settings()

# pydantic-settings는 os.environ에 값을 주입하지 않으므로,
# ChatOpenAI() 등 환경변수를 직접 읽는 라이브러리를 위해 수동 export
import os
if settings.OPENAI_API_KEY:
    os.environ.setdefault("OPENAI_API_KEY", settings.OPENAI_API_KEY)
if settings.PINECONE_API_KEY:
    os.environ.setdefault("PINECONE_API_KEY", settings.PINECONE_API_KEY)
