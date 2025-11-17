import gradio as gr
import requests
import json
import base64
from PIL import Image
import io
import numpy as np
import cv2
import os
# --- API 통신에 필요한 함수들 ---

WEBUI_URL = "http://127.0.0.1:7860"

def pil_to_base64(pil_image):
    with io.BytesIO() as stream:
        pil_image.save(stream, "PNG", pnginfo=None)
        return base64.b64encode(stream.getvalue()).decode('utf-8')

def image_file_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# --- 백엔드 핵심 로직 ---

def generate_art(sketch_image, prompt_text, negative_prompt, guidance_scale, steps):
    """
    스케치와 프롬프트를 받아 ControlNet(Canny)으로 이미지를 생성합니다.
    """
    
    if sketch_image is None:
        raise gr.Error("스케치 이미지를 업로드해야 합니다!")
    if not prompt_text:
        raise gr.Error("프롬프트를 입력해야 합니다!")

    print(f"'{prompt_text[:20]}...' 프롬프트로 이미지 생성 요청 받음...")

    # --- 스케치에서 Canny 외곽선 추출 ---
    print("스케치에서 Canny 외곽선 추출 중...")
    # PIL 이미지를 512x512로 리사이즈하고 OpenCV(numpy) 형식으로 변환
    sketch_image_resized = sketch_image.resize((512, 512))
    image_np = np.array(sketch_image_resized)
    
    # Canny 알고리즘 실행
    canny_np = cv2.Canny(image_np, 100, 200)
    
    # ControlNet에 보낼 수 있도록 다시 PIL 이미지로 변환
    canny_image_pil = Image.fromarray(canny_np)
    
    # Base64로 인코딩
    canny_base64 = pil_to_base64(canny_image_pil)
    print("Canny 외곽선 추출 및 인코딩 완료.")

    # --- AI 엔진(A1111)에 보낼 payload 작성 ---
    # (txt2img + ControlNet을 사용합니다)
    payload = {
        "prompt": prompt_text,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "width": 512,
        "height": 512,
        "cfg_scale": guidance_scale,
        "sampler_name": "Euler a",
        "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        "image": canny_base64, # <-- 추출한 Canny 이미지
                        "module": "none",      # <-- 'none' (이미 Canny 처리함)
                        "model": "control_v11p_sd15_canny [d14c016b]",
                        "weight": 1.0,
                        "control_mode": "ControlNet is more important" # (0: Balanced, 1: Prompt more important, 2: ControlNet more important)
                    }
                ]
            }
        }
    }

    try:
        # --- '엔진'에 이미지 생성 요청 (txt2img 엔드포인트) ---
        print("엔진(127.0.0.1:7860)에 [txt2img] + ControlNet 요청 전송 중...")
        # 10분(600초) 대기
        response = requests.post(url=f'{WEBUI_URL}/sdapi/v1/txt2img', json=payload, timeout=600) 
        response.raise_for_status() 

        r = response.json()

        if 'images' in r and len(r['images']) > 0:
            image_data = base64.b64decode(r['images'][0])
            result_image = Image.open(io.BytesIO(image_data))
            
            print("엔진으로부터 이미지 수신 완료. 프론트엔드로 반환합니다.")
            # Canny 이미지도 함께 반환하여 사용자가 비교
            return result_image, canny_image_pil
        else:
            print("API 응답에 이미지가 없습니다:", r)
            raise gr.Error("AI 엔진이 이미지를 반환하지 못했습니다. A1111 터미널을 확인하세요.")

    except requests.exceptions.Timeout:
        print("API 요청 시간 초과 (Timeout)")
        raise gr.Error("생성 시간이 너무 오래 걸려 중단되었습니다. A1111 엔진 상태를 확인하세요.")
    except requests.exceptions.RequestException as e:
        print(f"API 요청 중 치명적 오류 발생: {e}")
        raise gr.Error(f"A1111 엔진 연결 실패! (주소: {WEBUI_URL}). 엔진이 켜져 있는지 확인하세요.")

# --- 프론트엔드 UI (팀원 작업 공간) ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎨 AI 컨셉 아트 어시스턴트 (Sketch-to-Image)")
    
    gr.Markdown("1. 간단한 스케치를 업로드하세요.\n2. 생성할 프롬프트를 입력하세요.")
    with gr.Row():
        with gr.Column(scale=1):
            sketch_img = gr.Image(type="pil", label="스케치 이미지 (Sketch)", source="upload", tool="sketch")
            prompt_txt = gr.Textbox(label="긍정 프롬프트 (Prompt)", lines=3, placeholder="a knight in black iron plate armor, ...")
            neg_prompt_txt = gr.Textbox(label="부정 프롬프트 (Negative Prompt)", lines=3, placeholder="blurry, low quality, ...")
            with gr.Accordion("고급 설정", open=False):
                cfg_slider = gr.Slider(minimum=1, maximum=20, value=7.5, step=0.5, label="Guidance Scale (CFG)")
                steps_slider = gr.Slider(minimum=10, maximum=50, value=30, step=1, label="Steps")
            
            gen_btn = gr.Button("생성하기 (Generate)", variant="primary")

        with gr.Column(scale=1):
            result_img = gr.Image(label="결과 이미지 (Result)")
            canny_preview = gr.Image(label="추출된 외곽선 (Canny Preview)")

    gen_btn.click(
        fn=generate_art, 
        inputs=[sketch_img, prompt_txt, neg_prompt_txt, cfg_slider, steps_slider], 
        outputs=[result_img, canny_preview]
    )
    
    # (예시 기능을 사용하려면 'test_knight.jpg' 파일이 app.py와 같은 폴더에 있어야 합니다)
    gr.Examples(
        examples=[
            ["test_knight.jpg", "a knight in shining armor, detailed metal plates, holding a sword, fantasy art", "blurry, low quality, deformed", 7.5, 30],
            ["test_knight.jpg", "dark knight, black rusty armor, glowing red eyes, fantasy art", "blurry, low quality, shining, silver", 7.5, 30]
        ],
        inputs=[sketch_img, prompt_txt, neg_prompt_txt, cfg_slider, steps_slider]
    )

# --- 5. '매니저 API' 서버 실행 ---
print("AI 컨셉 아트 어시스턴트 서버(127.0.0.1:8000)를 시작합니다...")
# A1111 API와 포트가 겹치지 않도록 8000번 포트 사용
demo.launch(server_name="0.0.0.0", server_port=8000, share=True)