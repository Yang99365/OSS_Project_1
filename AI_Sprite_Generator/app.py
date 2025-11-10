import gradio as gr
import requests
import json
import base64
from PIL import Image
import io

# --- API 통신에 필요한 함수들 ---

WEBUI_URL = "http://127.0.0.1:7860"

def pil_to_base64(pil_image):
    with io.BytesIO() as stream:
        pil_image.save(stream, "PNG", pnginfo=None)
        return base64.b64encode(stream.getvalue()).decode('utf-8')

def image_file_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# --- 백엔드 핵심 로직 (님 작업 공간) ---

def generate_sprite(character_image, motion_type):
    """
    [최종 결합 버전]
    원본 이미지(img2img)의 스타일을 유지하면서
    ControlNet 포즈를 강력하게 적용합니다.
    """
    
    print(f"'{motion_type}' 모션 생성 요청 받음... [최종 버전]")

    # TODO: 지금은 테스트를 위해 '걷기' 모션만 가정합니다.
    pose_image_path = "poses/walk_01.png" # 테스트용 포즈 이미지 경로

    # 1. img2img payload (GPU 최종 버전)
    payload = {
        "init_images": [ pil_to_base64(character_image) ], # 원본 캐릭터 이미지
        "prompt": "1 character, full body, best quality, solo, white background",
        "negative_prompt": "monochrome, lowres, bad anatomy, worst quality, blurry",
        "steps": 20,
        "width": 512,
        "height": 512,
        "cfg_scale": 7,
        "denoising_strength": 0.75, # 포즈를 바꿀 수 있도록 넉넉한 자유도 부여
        "sampler_name": "Euler a",
        "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        "image": image_file_to_base64(pose_image_path),
                        "module": "none",
                        "model": "control_v11p_sd15_openpose [cab727d4]", # <-- 우리가 찾아낸 정확한 모델 이름
                        "weight": 1.0,
                        "control_mode": "ControlNet is more important" # <-- ControlNet 최우선 적용
                    }
                ]
            }
        }
    }

    try:
        # 2. '엔진'에 이미지 생성 요청 (img2img 엔드포인트)
        print("엔진(127.0.0.1:7860)에 [img2img] 요청 전송 중...")
        # 10분(600초) 대기
        response = requests.post(url=f'{WEBUI_URL}/sdapi/v1/img2img', json=payload, timeout=600) 
        response.raise_for_status() 

        r = response.json()

        if 'images' in r and len(r['images']) > 0:
            image_data = base64.b64decode(r['images'][0])
            result_image = Image.open(io.BytesIO(image_data))
            
            print("엔진으로부터 이미지 수신 완료. 프론트엔드로 반환합니다.")
            return result_image
        else:
            print("API 응답에 이미지가 없습니다:", r)
            return None

    except requests.exceptions.RequestException as e:
        print(f"API 요청 중 치명적 오류 발생: {e}")
        return None

# --- 프론트엔드 UI (팀원 작업 공간) ---

with gr.Blocks() as demo:
    gr.Markdown("# 🤖 AI 스프라이트 시트 생성기 (GPU Ver.)")
    with gr.Row():
        char_img = gr.Image(type="pil", label="캐릭터 이미지 업로드")
        motion = gr.Dropdown(choices=["걷기", "달리기", "점프"], label="모션 선택")
        output_img = gr.Image(label="결과 이미지")
    
    btn = gr.Button("생성하기!")
    btn.click(fn=generate_sprite, inputs=[char_img, motion], outputs=[output_img])

# --- '매니저 API' 서버 실행 ---

print("매니저 API 서버(127.0.0.1:8000)를 시작합니다...")
# share=True는 집에서만 테스트할 땐 꺼도 됩니다.
demo.launch(server_name="0.0.0.0", server_port=8000, share=False)