import sys
import os
import toml
import google.generativeai as genai

# --- 설정 ---
CONFIG_PATH = os.path.expanduser("~/.config/gemini-cli.toml")

def get_api_key():
    try:
        config = toml.load(CONFIG_PATH)
        return config['gemini'].get('token') or config['gemini'].get('api_key')
    except Exception:
        return None

def chat(prompt):
    api_key = get_api_key()
    if not api_key:
        print(f"❌ 설정 파일({CONFIG_PATH})에서 API 키를 찾을 수 없습니다.")
        return

    genai.configure(api_key=api_key)
    
    # ★ 여기가 핵심 변경사항입니다 ★
    # 사용자님 목록에 있는 최신 모델들을 순서대로 시도합니다.
    models_to_try = [
        'gemini-2.5-flash',       # 최신 2.5 버전 (가장 추천)
        'gemini-2.0-flash',       # 2.0 버전
        'gemini-flash-latest',    # 최신 플래시 자동 선택
        'gemini-1.5-flash'        # 구버전 (예비용)
    ]
    
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            print(f"\n🤖 Gemini ({model_name}):\n{response.text}")
            return
        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower():
                continue # 다음 모델 시도
            else:
                print(f"❌ Error with {model_name}: {e}")
                return

    print("\n❌ 사용 가능한 모델을 찾지 못했습니다.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
        chat(user_input)
    else:
        print("사용법: python gemini_run.py [질문]")