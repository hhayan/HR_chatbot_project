# frontend/app.py
"""
Streamlit Frontend — HR Chatbot UI
- 순수 UI 로직만 담당
- API 키를 절대 보유하지 않음 (보안 원칙)
- FastAPI 백엔드에 HTTP 요청으로만 통신
"""

import os
import requests
import streamlit as st

# 백엔드 URL (로컬: http://localhost:8000 / Docker: http://backend:8000)
BACKEND_URL = os.getenv("BACKEND_URL", "https://hr-chatbot-project.onrender.com")


def send_message(query: str) -> str:
    """백엔드 /chat 엔드포인트에 사용자 질문 전송"""
    try:
        res = requests.post(
            f"{BACKEND_URL}/chat",
            json={"query": query},
            timeout=120,
        )
        res.raise_for_status()
        return res.json().get("answer", "응답을 받지 못했습니다.")
    except requests.exceptions.ConnectionError:
        return "⚠️ 백엔드 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요."
    except requests.exceptions.Timeout:
        return "⚠️ 요청 시간이 초과되었습니다. 잠시 후 다시 시도해주세요."
    except requests.exceptions.RequestException as e:
        return f"⚠️ 서버 오류: {e}"


# ==================== 페이지 설정 ====================
st.set_page_config(
    page_title="가이다 HR 챗봇",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 가이다 HR 챗봇")
st.caption("가이다 플레이 스튜디오(GPS) 사내 HR 정책 안내 챗봇입니다.")


# ==================== 세션 상태 초기화 ====================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! HR 관련 궁금한 점이 있으면 편하게 질문해주세요. 😊"}
    ]


# ==================== 대화 이력 렌더링 ====================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ==================== 사용자 입력 및 응답 처리 ====================
if prompt := st.chat_input("HR 관련 질문을 입력하세요"):
    # 사용자 메시지 추가 & 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 백엔드 호출 & 응답 표시
    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            answer = send_message(prompt)
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
