from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from diffusers import StableDiffusionPipeline
from PIL import Image
import torch
import io

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
    업로드 이미지는 사용하지 않음
    """

    result = pipe(
        prompt=prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
    )

    img = result.images[0]

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")
