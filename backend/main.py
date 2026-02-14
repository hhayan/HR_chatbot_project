# backend/main.py
"""
FastAPI Backend — HR Chatbot API Server
- LangGraph StateGraph를 사용한 RAG 파이프라인
- OpenAI / Pinecone 키는 이 서버에서만 관리 (보안 원칙)
- POST /chat 엔드포인트 제공
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from config import settings, ENV_PATH
from graph import graph


# ==================== Schemas ====================
class ChatRequest(BaseModel):
    """채팅 요청"""
    query: str


class ChatResponse(BaseModel):
    """채팅 응답"""
    answer: str


# ==================== FastAPI App ====================
app = FastAPI(
    title="가이다 HR 챗봇 API",
    description="RAG 기반 사내 HR 질문 응답 시스템",
    version="1.0.0",
)

# CORS — Streamlit 프론트엔드 허용
origins = settings.ALLOWED_ORIGINS.split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    """서버 시작 시 Settings 로딩 확인"""
    """서버 시작 시 Settings 로딩 확인"""
    print(f"[Settings] .env 경로 = {ENV_PATH}")


# ==================== Endpoints ====================
@app.get("/health")
def health_check():
    """서버 상태 확인"""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    사용자 질문을 받아 LangGraph 그래프를 실행하고 답변을 반환합니다.
    """
    result = graph.invoke({"messages": [("human", req.query)]})
    answer = result.get("final_answer", "답변을 생성하지 못했습니다.")
    return ChatResponse(answer=answer)
