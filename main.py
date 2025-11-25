# server/main.py

from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from diffusers import StableDiffusionPipeline
from PIL import Image
import torch
import io
import os

from dotenv import load_dotenv
from openai import OpenAI

# -------------------------
# 🔑 환경변수 로드 & OpenAI 클라이언트
# -------------------------
load_dotenv()  # .env에서 OPENAI_API_KEY 읽기

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")  # 절대 코드에 키 직접 쓰지 말기!
)


def refine_prompt(user_prompt: str) -> str:
    """
    OpenAI(GPT-4.1)를 사용해서 Stable Diffusion용 고퀄 프롬프트로 확장/보정
    실패하면 원래 프롬프트 그대로 반환
    """
    if not user_prompt:
        return "a high quality 2D game character illustration"

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "너는 Stable Diffusion 프롬프트 최적화 전문가다. "
                        "사용자가 적은 짧은 설명을, "
                        "고품질 2D 게임 그래픽(일러스트, 스프라이트)에 적합한 "
                        "영어 프롬프트로 구체적으로 확장해라. "
                        "스타일, 조명, 구도, 화질 등을 자세히 써라."
                    ),
                },
                {
                    "role": "user",
                    "content": f"다음 설명을 Stable Diffusion용 영어 프롬프트로 바꿔줘: {user_prompt}",
                },
            ],
            max_tokens=200,
        )
        refined = resp.choices[0].message.content.strip()
        return refined or user_prompt
    except Exception as e:
        print("[OpenAI 프롬프트 보정 오류]", e)
        return user_prompt


# -------------------------
# FastAPI 기본 설정
# -------------------------

app = FastAPI()

# CORS : Streamlit에서 API 호출 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_ID = "runwayml/stable-diffusion-v1-5"

print("🔹 Stable Diffusion txt2img 모델 로드 중...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32
).to("cpu")
print("✅ txt2img 모델 로드 완료!")


@app.post("/generate")
async def generate_image(prompt: str = Form(...)):
    """
    Stable Diffusion txt2img 전용 엔드포인트
    - 업로드 이미지는 절대 사용 안 함
    - 프롬프트는 먼저 OpenAI로 보정한 후 SD에 전달
    """

    # 1) OpenAI로 프롬프트 보정
    refined_prompt = refine_prompt(prompt)
    print("🧠 원본 프롬프트 :", prompt)
    print("✨ 보정된 프롬프트 :", refined_prompt)

    # 2) Stable Diffusion txt2img 호출
    result = pipe(
        prompt=refined_prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
    )

    img = result.images[0]

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")
