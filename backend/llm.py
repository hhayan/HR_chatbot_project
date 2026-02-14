# backend/llm.py
"""
LLM 팩토리 함수 — 노드별 적합한 모델 반환
환경 변수는 config.settings에서 중앙 관리
"""

from langchain_openai import ChatOpenAI
from config import settings


def get_llm(role: str = "gen") -> ChatOpenAI:
    """
    노드별로 적합한 LLM 모델을 반환하는 팩토리 함수

    Args:
        role (str): 역할별 모델 선택
            - "gen": 본문 생성/분석
            - "router": 라우터 분기 판단

    Returns:
        ChatOpenAI: 설정된 LLM 인스턴스
    """
    api_key = settings.OPENAI_API_KEY
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

    # 역할별 고정 모델 매핑
    model_map = {
        "gen": settings.GEN_LLM,
        "router1": settings.ROUTER1_LLM,
        "router2": settings.ROUTER2_LLM,
        "router": settings.ROUTER2_LLM,  # 하위 호환
    }

    model_name = model_map.get(role, model_map["gen"])

    try:
        return ChatOpenAI(
            model=model_name,
            temperature=0,
            api_key=api_key,
        )
    except Exception as e:
        raise RuntimeError(f"LLM 초기화 실패 (role: {role}, model: {model_name}): {e}")