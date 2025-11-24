import streamlit as st
import requests
from PIL import Image
import io

API_URL = "http://127.0.0.1:8000/generate"

# -------------------------
# ⭐ CSS 파일 불러오는 함수
# -------------------------
def load_css():
    try:
        with open("style.css", "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        pass  # style.css 없으면 그냥 무시됨

# CSS 적용
load_css()

# -------------------------
# UI 시작
# -------------------------
st.set_page_config(page_title="AI 이미지 바리에이션 웹", layout="centered")

st.title("🎨 AI 이미지 생성기")
st.markdown("두 가지 모드 중 하나를 선택하세요:")

# -------------------------
# ⭐ 사용 모드 선택
# -------------------------
mode = st.radio(
    "사용할 모드 선택",
    ("프롬프트로 이미지 생성", "이미지 파일 사용하기"),
    horizontal=True
)




# -------------------------
# 🌈 1) 프롬프트 기반 생성 모드
# -------------------------
if mode == "프롬프트로 이미지 생성":

    st.subheader("✨ 프롬프트로 AI 이미지 생성하기")

    prompt = st.text_input("Chat bot에게 생성할 이미지를 입력해주세요.")

    if st.button("이미지 생성하기"):
        if not prompt:
            st.warning("생성할 이미지를 입력해주세요.")
        else:
            with st.spinner("이미지 생성 중..."):
                response = requests.post(API_URL, data={"prompt": prompt})

                if response.status_code == 200:
                    img = Image.open(io.BytesIO(response.content))
                    st.image(img, caption="AI 생성 이미지", use_column_width=True)

                    st.download_button(
                        label="📥 이미지 다운로드",
                        data=response.content,
                        file_name="generated.png",
                        mime="image/png"
                    )
                else:
                    st.error("이미지 생성 실패. 서버 로그를 확인하세요.")


# -------------------------
# 📁 2) 업로드 이미지 사용 모드
# -------------------------
else:
    st.subheader("📁 업로드한 이미지 사용하기")

    uploaded = st.file_uploader(
        "이미지를 업로드하세요 (PNG / JPG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="업로드한 이미지", use_column_width=True)

        # 다운로드 기능
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        st.download_button(
            label="📥 업로드 이미지 다운로드",
            data=buffer.getvalue(),
            file_name="uploaded.png",
            mime="image/png"
        )
