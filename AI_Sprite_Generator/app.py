import gradio as gr
import requests
import json
import base64
from PIL import Image, ImageFilter # ★ ImageFilter 필수 포함
import io
import numpy as np
import cv2
import os
from dotenv import load_dotenv
from openai import OpenAI

# --- 설정 (서버 주소가 바뀌면 여기를 수정하세요) ---
WEBUI_URL = ""
CONTROLNET_MODEL_NAME = "kohya_controllllite_xl_canny [2ed264be]"

# 환경변수 로드
load_dotenv()
# OpenAI 클라이언트 초기화 (API Key는 .env 파일에 있어야 함)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# --- 유틸리티 함수 ---r
def pil_to_base64(pil_image):
    with io.BytesIO() as stream:
        pil_image.save(stream, "PNG", pnginfo=None)
        return base64.b64encode(stream.getvalue()).decode('utf-8')

def refine_prompt(user_prompt: str) -> str:
    """
    OpenAI(GPT-4)를 사용해서 Stable Diffusion용 고퀄 프롬프트로 확장/보정
    실패하면 원래 프롬프트 그대로 반환
    (참고: friend)main.py에서 가져옴)
    """
    if not user_prompt:
        return "a high quality 2D game character illustration"

    # Pony/SDXL 스타일에 맞게 시스템 메시지 변경
    system_message = (
        "너는 SDXL Pony 모델 프롬프트 최적화 전문가다. "
        "사용자가 적은 짧은 설명을, "
        "고품질 애니메이션/일러스트 스타일 영어 프롬프트로 구체적으로 확장해라. "
        "스타일, 조명, 구도, 화질 등을 자세히 써라. 답변은 오직 영어로만 해라."
    )
    
    try:
        print(f"[OpenAI 요청] 프롬프트 보정: {user_prompt}")
        resp = client.chat.completions.create(
            model="gpt-4o-mini", # 더 빠르고 저렴한 모델로 변경 (선택 사항)
            messages=[
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": f"다음 설명을 Stable Diffusion용 영어 프롬프트로 바꿔줘: {user_prompt}",
                },
            ],
            max_tokens=250,
        )
        refined = resp.choices[0].message.content.strip()
        
        # 퀄리티 태그는 이미 process_pony_prompt에서 추가하므로 여기선 제거
        # clean_refined = refined.replace("score_9, score_8_up, score_7_up, score_6_up, source_anime, high quality, ", "").strip()
        
        print(f"✨ 보정된 프롬프트 : {refined}")
        return refined or user_prompt
    except Exception as e:
        print(f"[OpenAI 프롬프트 보정 오류] {e}")
        return user_prompt
    
# --- Pony/SDXL 전용 프롬프트 처리 ---
def process_pony_prompt(user_prompt, negative_prompt):
    # Pony 모델은 퀄리티 태그가 필수입니다.
    quality_tags = "score_9, score_8_up, score_7_up, score_6_up, source_anime, high quality, "
    full_prompt = quality_tags + user_prompt
    
    # Pony 권장 부정 프롬프트
    base_negative = "score_4, score_5, score_6, low quality, bad anatomy, worst quality, text, watermark, "
    full_negative = base_negative + negative_prompt
    
    return full_prompt, full_negative

# --- 핵심 로직 ---

def generate_canny(sketch_dict, prompt_text, negative_prompt, guidance_scale, steps):
    """
    스케치 탭: 깨끗한 원본으로 선을 따고, 칠한 색감을 입힙니다.
    """
    if sketch_dict is None:
        raise gr.Error("이미지를 업로드해주세요!")

    # 1. 소스 분리 (핵심!)
    # background: 사용자가 업로드한 원본 스케치 (선이 선명함 -> ControlNet용)
    # composite: 사용자가 브러시로 색칠한 결과물 (색이 있음 -> img2img용)
    
    clean_line_art = sketch_dict["background"]
    colored_draft = sketch_dict["composite"]
    
    # 예외처리: 이미지를 업로드 안 하고 빈 캔버스에 바로 그렸을 경우
    # background가 없으므로 어쩔 수 없이 composite을 사용
    if not clean_line_art:
        clean_line_art = colored_draft

    if colored_draft is None:
         raise gr.Error("스케치를 그려주세요!")
    
    # SDXL 표준 해상도
    width, height = 1024, 1024
    
    # 리사이징
    clean_resized = clean_line_art.resize((width, height))
    colored_resized = colored_draft.resize((width, height))
    
    # 2. Canny 추출 (깨끗한 원본 스케치 사용)
    # 이제 색칠을 아무리 개판으로 해도, 원본 선이 살아있으므로 형태가 유지됩니다.
    image_np = np.array(clean_resized)
    
    # 알파 채널 제거 (투명 배경일 경우 검은 배경으로 변환 방지용 등)
    if image_np.shape[-1] == 4:
         image_np = image_np[:, :, :3]
         
    canny_np = cv2.Canny(image_np, 50, 100) 
    canny_image_pil = Image.fromarray(canny_np)

    # 3. Base64 인코딩
    # img2img에는 '색칠된 버전'을 보냅니다.
    init_base64 = pil_to_base64(colored_resized)
    # ControlNet에는 '깨끗한 선 버전'을 보냅니다.
    canny_base64 = pil_to_base64(canny_image_pil)

    # 4. 프롬프트 강화
    final_prompt, final_negative = process_pony_prompt(prompt_text, negative_prompt)

    # 5. API 요청 Payload
    payload = {
        "prompt": final_prompt,
        "negative_prompt": final_negative,
        "init_images": [init_base64], 
        "steps": steps,
        "width": width,
        "height": height,
        "cfg_scale": guidance_scale,
        "sampler_name": "Euler a",
        
        # 색감은 반영하되 형태는 ControlNet이 꽉 잡아야 하므로 Denoising을 높여도 됩니다.
        "denoising_strength": 0.85, 
        
        "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        "image": canny_base64,
                        "module": "none", 
                        "model": CONTROLNET_MODEL_NAME,
                        "weight": 1.2,               # 가중치를 1.0 -> 1.2로 강화
                        "control_mode": "ControlNet is more important", # 프롬프트보다 스케치 우선
                    }
                ]
            }
        }
    }

    try:
        print("[요청] 스케치 -> 이미지 (구도 유지 강화) 요청 중...")
        response = requests.post(url=f'{WEBUI_URL}/sdapi/v1/img2img', json=payload, timeout=600)
        response.raise_for_status()
        r = response.json()
        if 'images' in r:
            return Image.open(io.BytesIO(base64.b64decode(r['images'][0]))), canny_image_pil
            
    except Exception as e:
        print(f"[오류] {e}")
        raise gr.Error(f"오류 발생: {e}")

def generate_inpaint(image_editor_dict, prompt_text, negative_prompt, guidance_scale, steps):
    """
    인페인팅 탭: 마스킹된 영역만 수정합니다.
    """
    # 1. 입력 확인
    if image_editor_dict is None:
        raise gr.Error("이미지를 업로드하고 마스킹을 해주세요!")

    # 2. 이미지 및 마스크 추출
    # background: 원본 이미지
    # layers: 사용자가 칠한 마스크 (투명 배경에 칠한 부분만 불투명)
    init_img = image_editor_dict["background"]
    
    # 마스크 처리 (layers 리스트의 첫 번째 레이어 사용)
    if not image_editor_dict["layers"]:
        raise gr.Error("마스크 영역이 감지되지 않았습니다. 수정할 곳을 색칠해주세요.")
        
    mask_layer = image_editor_dict["layers"][0] # RGBA 이미지
    
    # 마스크 생성: 알파 채널을 추출하여 흑백(Binary) 마스크로 변환
    # 칠한 부분(불투명) -> 흰색(255), 안 칠한 부분(투명) -> 검은색(0)
    mask_np = np.array(mask_layer)
    if mask_np.shape[2] == 4: # 알파 채널이 있다면
        alpha_channel = mask_np[:, :, 3]
        mask_image = Image.fromarray(alpha_channel).convert("L")
    else:
        # 혹시라도 알파 채널이 없으면 그레이스케일 변환
        mask_image = mask_layer.convert("L")

    # SDXL 해상도 리사이징 (1024x1024 권장)
    width, height = 1024, 1024
    init_img_resized = init_img.resize((width, height))
    mask_img_resized = mask_image.resize((width, height))

    # 3. Base64 인코딩
    init_base64 = pil_to_base64(init_img_resized)
    mask_base64 = pil_to_base64(mask_img_resized)

    # 4. 프롬프트 강화 (Pony 전용)
    final_prompt, final_negative = process_pony_prompt(prompt_text, negative_prompt)

    # 5. API 페이로드 구성 (img2img 엔드포인트 사용)
    payload = {
        "prompt": final_prompt,
        "negative_prompt": final_negative,
        "init_images": [init_base64],
        "mask": mask_base64,
        "steps": steps,
        "width": width,
        "height": height,
        "cfg_scale": guidance_scale,
        "sampler_name": "Euler a",
        
        # --- 인페인팅 핵심 파라미터 ---
        "mask_blur": 4,             # 마스크 경계 부드럽게 (자연스러운 합성을 위해 필수)
        "inpainting_fill": 1,       # 1 = 원본 유지(Original). 수정하려는 내용이 기존 색감과 비슷하면 1, 아예 다르면 0(채우기)이나 2(노이즈) 사용
        "inpaint_full_res": True,   # True = 마스크 영역만 고화질로 다시 그림 (얼굴/디테일 수정 시 필수)
        "inpaint_full_res_padding": 32, # 주변 영역 참조 픽셀 수
        "denoising_strength": 0.75, # 0.75 = 기존 형태를 많이 바꾸면서 생성. (0.4 이하는 거의 안 바뀜)
        "resize_mode": 0            # 0 = Just resize
    }

    try:
        print("[요청] 이미지 -> 이미지 (Inpaint) 요청 중...")
        # txt2img가 아니라 img2img 엔드포인트를 사용합니다.
        response = requests.post(url=f'{WEBUI_URL}/sdapi/v1/img2img', json=payload, timeout=600)
        response.raise_for_status()
        r = response.json()
        
        if 'images' in r:
            result_img = Image.open(io.BytesIO(base64.b64decode(r['images'][0])))
            return result_img
            
    except Exception as e:
        print(f"[오류] {e}")
        raise gr.Error(f"인페인팅 오류 발생: {e}")

# --- UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🦄 AI 아트 어시스턴트 (SDXL Pony Edition)")
    gr.Markdown("⚠️ **주의:** 친구 서버에 **Hyper3D 모델**이 로드되어 있어야 하고, **SDXL용 ControlNet**이 설치되어 있어야 합니다.")
    
    with gr.Tabs():
        with gr.TabItem("스케치 완성 (Sketch)"):
            with gr.Row():
                sketch = gr.ImageEditor(type="pil", label="스케치", height=600)
                with gr.Column():
                    prompt = gr.Textbox(label="프롬프트", placeholder="1girl, silver armor, knight...")
                    neg = gr.Textbox(label="부정 프롬프트", placeholder="extra fingers...")
                    btn = gr.Button("생성 (Generate)", variant="primary")
                    result = gr.Image(label="결과")
                    debug = gr.Image(label="Canny 미리보기", height=200)
            btn.click(generate_canny, [sketch, prompt, neg, gr.Number(7.0, visible=False), gr.Number(30, visible=False)], [result, debug])

        with gr.TabItem("부분 수정 (Inpaint)"):
             with gr.Row():
                edit_img = gr.ImageEditor(type="pil", label="마스킹 (창 그릴 곳만 칠하세요!)", height=600)
                with gr.Column():
                    i_prompt = gr.Textbox(label="수정 내용 (예: red spear, glowing lance)")
                    i_neg = gr.Textbox(label="부정 프롬프트")
                    i_btn = gr.Button("수정 (Inpaint)", variant="primary")
                    i_result = gr.Image(label="결과")
             i_btn.click(generate_inpaint, [edit_img, i_prompt, i_neg, gr.Number(7.0, visible=False), gr.Number(40, visible=False)], [i_result])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8000, share=True)