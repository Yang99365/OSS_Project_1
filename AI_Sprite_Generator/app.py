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

def generate_canny(sketch_dict, prompt_text, negative_prompt, guidance_scale, steps):
    """ 스케치 이미지를 받아 Canny 외곽선을 추출하고 ControlNet으로 이미지를 생성합니다. """
    
    if sketch_dict is None or sketch_dict["composite"] is None:
        raise gr.Error("스케치 이미지를 업로드하거나 그려야 합니다!")

    sketch_image = sketch_dict["composite"]

    # 1. Canny 외곽선 추출
    image_np = np.array(sketch_image.resize((512, 512)))
    canny_np = cv2.Canny(image_np, 100, 200)
    canny_image_pil = Image.fromarray(canny_np)
    canny_base64 = pil_to_base64(canny_image_pil)

    # 2. API Payload 작성 (txt2img + ControlNet Canny)
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
                        "image": canny_base64,
                        "module": "none",
                        "model": "control_v11p_sd15_canny [d14c016b]", # ★★★ A1111 모델명으로 변경할 것! ★★★
                        "weight": 1.0,
                        "control_mode": "ControlNet is more important"
                    }
                ]
            }
        }
    }

    try:
        # 3. 이미지 생성 요청 (txt2img 엔드포인트)
        response = requests.post(url=f'{WEBUI_URL}/sdapi/v1/txt2img', json=payload, timeout=600) 
        response.raise_for_status() 
        r = response.json()

        if 'images' in r and len(r['images']) > 0:
            image_data = base64.b64decode(r['images'][0])
            result_image = Image.open(io.BytesIO(image_data))
            return result_image, canny_image_pil
        else:
            raise gr.Error("AI 엔진이 이미지를 반환하지 못했습니다. A1111 터미널을 확인하세요.")

    except requests.exceptions.RequestException as e:
        raise gr.Error(f"API 요청 중 치명적 오류 발생: 엔진 연결 실패 또는 타임아웃. A1111 엔진이 '--api' 옵션으로 켜져 있는지 확인하세요.")

def generate_inpaint(image_editor_dict, prompt_text, negative_prompt, guidance_scale, steps):
    """ [성공 로직] 마스킹된 영역을 Inpainting으로 수정합니다. (ControlNet Canny와 결합) """
    
    # Gradio 입력 검증
    if image_editor_dict is None: raise gr.Error("이미지를 업로드해야 합니다!")
    image = image_editor_dict.get("background")
    mask_layers = image_editor_dict.get("layers")
    if image is None: raise gr.Error("원본 이미지가 없습니다!")
    if mask_layers is None or len(mask_layers) == 0 or mask_layers[0] is None:
        raise gr.Error("수정할 영역을 브러시로 칠해야 합니다 (마스크가 없음)!")

    # 1. 이미지 및 마스크 준비 (512x512)
    original_image = image.resize((512, 512)).convert("RGB")
    mask = mask_layers[0].resize((512, 512))
    mask_alpha = mask.split()[-1] # 마스크의 Alpha 채널 (흑백)

    # 2. ControlNet 가이드 생성 (원본 캐릭터 형태 유지)
    image_to_canny_np = np.array(original_image)
    mask_alpha_np = np.array(mask_alpha)
    
    # 마스크 영역을 흰색(255)으로 칠하여, Canny가 창이 아닌 '캐릭터' 외곽선만 잡도록 유도
    image_to_canny_np[mask_alpha_np > 0] = [255, 255, 255] 
    
    canny_np = cv2.Canny(image_to_canny_np, 100, 200)
    canny_image_pil = Image.fromarray(canny_np).convert("RGB")
    canny_base64 = pil_to_base64(canny_image_pil)

    # 3. Inpaint가 채울 '수정 대상 이미지' 준비
    image_to_modify = np.array(original_image)
    image_to_modify[mask_alpha_np > 0] = [127, 127, 127] # 마스크 영역을 회색으로
    image_to_modify_pil = Image.fromarray(image_to_modify)
    
    # 4. API Payload 작성 (Inpaint API + ControlNet Canny)
    payload = {
        "prompt": prompt_text,
        "negative_prompt": negative_prompt,
        "steps": 40, 
        "width": 512,
        "height": 512,
        "cfg_scale": 9.0,
        "sampler_name": "Euler a",
        "mask_image": pil_to_base64(mask_alpha), 
        "init_images": [pil_to_base64(image_to_modify_pil)],
        "denoising_strength": 0.8,
        
        "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        "image": canny_base64,
                        "module": "none",
                        "model": "control_v11p_sd15_canny [d14c016b]", # ★★★ A1111 모델명으로 변경할 것! ★★★
                        "weight": 0.5, 
                        "control_mode": "Balanced"
                    }
                ]
            }
        }
    }

    try:
        # 5. 이미지 생성 요청 (img2img 엔드포인트)
        response = requests.post(url=f'{WEBUI_URL}/sdapi/v1/img2img', json=payload, timeout=600) 
        response.raise_for_status() 
        r = response.json()

        if 'images' in r and len(r['images']) > 0:
            image_data = base64.b64decode(r['images'][0])
            result_image = Image.open(io.BytesIO(image_data))
            return result_image
        else:
            raise gr.Error("AI 엔진이 이미지를 반환하지 못했습니다. A1111 터미널을 확인하세요.")

    except requests.exceptions.RequestException as e:
        raise gr.Error(f"API 요청 중 치명적 오류 발생: 엔진 연결 실패 또는 타임아웃. A1111 엔진이 '--api' 옵션으로 켜져 있는지 확인하세요.")

# --- 프론트엔드 UI (테스트용 임시 Gradio) ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎨 AI 게임 리소스 어시스턴트 (2-in-1)")
    
    with gr.Tabs():
        
        # --- 탭 1: Sketch-to-Image ---
        with gr.TabItem("1. Sketch-to-Image (Canny)"):
            gr.Markdown("**기능:** 스케치를 업로드하고 프롬프트를 입력하면, AI가 스케치에 맞춰 채색/완성합니다.")
            with gr.Row():
                with gr.Column(scale=1):
                    c_sketch = gr.ImageEditor(type="pil", label="스케치 이미지 (Sketch)", value="test_knight.jpg")
                    c_prompt = gr.Textbox(label="긍정 프롬프트", placeholder="a knight in black iron plate armor, red plume...")
                    c_neg_prompt = gr.Textbox(label="부정 프롬프트", placeholder="blurry, low quality...")
                    c_scale = gr.Slider(minimum=1, maximum=20, value=7.5, step=0.5, label="Guidance Scale (CFG)")
                    c_steps = gr.Slider(minimum=10, maximum=50, value=30, step=1, label="Steps")
                    c_btn = gr.Button("생성하기 (Generate)", variant="primary")
                with gr.Column(scale=1):
                    c_result = gr.Image(label="결과 이미지 (Result)")
                    c_preview = gr.Image(label="추출된 외곽선 (Canny Preview)")
            
            c_btn.click(fn=generate_canny, inputs=[c_sketch, c_prompt, c_neg_prompt, c_scale, c_steps], outputs=[c_result, c_preview], show_progress='full')

        # --- 탭 2: Partial Edit ---
        with gr.TabItem("2. Partial Edit (Inpainting)"):
            gr.Markdown("**기능:** 이미지를 업로드하고 **직선 도구로 수정할 영역을 칠한(Masking)** 뒤, 프롬프트를 입력하면 해당 부분만 수정합니다.")
            with gr.Row():
                with gr.Column(scale=1):
                    i_img = gr.ImageEditor(type="pil", label="수정할 이미지 (Image & Mask)", value="test_knight.jpg")
                    i_prompt = gr.Textbox(label="긍정 프롬프트 (바꿀 내용)", placeholder="a iron spear, sharp tip, long pole")
                    i_neg_prompt = gr.Textbox(label="부정 프롬프트", placeholder="blurry, low quality, club, wooden stick, deformed, extra hands")
                    i_scale = gr.Slider(minimum=1, maximum=20, value=9.0, step=0.5, label="Guidance Scale (CFG)")
                    i_steps = gr.Slider(minimum=10, maximum=50, value=40, step=1, label="Steps")
                    i_btn = gr.Button("부분 수정 (Inpaint)", variant="primary")
                with gr.Column(scale=1):
                    i_result = gr.Image(label="결과 이미지 (Result)")
        
            i_btn.click(fn=generate_inpaint, inputs=[i_img, i_prompt, i_neg_prompt, i_scale, i_steps], outputs=[i_result], show_progress='full')


# --- 5. '매니저 API' 서버 실행 ---
print("AI 컨셉 아트 어시스턴트 서버(127.0.0.1:8000)를 시작합니다...")
# A1111 API와 포트가 겹치지 않도록 8000번 포트 사용
demo.launch(server_name="0.0.0.0", server_port=8000, share=False)