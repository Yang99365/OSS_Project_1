import gradio as gr
import requests  # API 요청을 위한 라이브러리
import json
import base64
from PIL import Image
import io

# WebUI API가 실행 중인 주소
WEBUI_URL = "http://127.0.0.1:7860"

# PIL 이미지를 Base64 문자열로 변환하는 함수
def pil_to_base64(pil_image):
    with io.BytesIO() as stream:
        pil_image.save(stream, "PNG", pnginfo=None)
        return base64.b64encode(stream.getvalue()).decode('utf-8')

# 이미지 파일 경로를 Base64 문자열로 변환하는 함수
def image_file_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# --- 백엔드 핵심 로직 ---

def generate_sprite(character_image, motion_type):
    """
    이 함수가 '매니저'의 핵심 임무입니다.
    프론트엔드에서 이미지와 모션 타입을 받아서,
    WebUI 엔진에 요청을 보내고, 결과 이미지를 반환합니다.
    """
    
    print(f"'{motion_type}' 모션 생성 요청 받음...")

    # TODO: 지금은 테스트를 위해 '걷기' 모션만 가정하고,
    # 'poses/walk_01.png'라는 하나의 포즈 파일만 사용합니다.
    # 나중에 이 부분을 'motion_type'에 따라 여러 포즈를 반복하도록 수정해야 합니다.
    pose_image_path = "poses/walk_01.png" # 테스트용 포즈 이미지 경로

    # 1. WebUI 엔진(:7860)에 보낼 payload(명령서)를 작성합니다.
    #    'img2img' API를 사용하여 원본 캐릭터의 스타일을 유지합니다.
    payload = {
        "init_images": [ pil_to_base64(character_image) ], # 원본 캐릭터 이미지
        "prompt": "1 character, full body, best quality, solo",
        "negative_prompt": "monochrome, lowres, bad anatomy, worst quality, blurry",
        "steps": 20,
        "width": 512,
        "height": 512,
        "cfg_scale": 7,
        "denoising_strength": 0.75, # 원본 이미지에서 얼마나 많이 바꿀지
        "sampler_name": "Euler a",
        "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        "image": image_file_to_base64(pose_image_path),
                        "module": "none", # <-- "추출하지 말고, 이 뼈대 그대로"
                        "model": "control_v11p_sd15_openpose [cab7509c]",
                        "weight": 1.0,
                        "control_mode": "ControlNet is more important"
                    }
                ]
            }
        }
    }

    try:
        # 2. '엔진'에 이미지 생성 요청을 보냅니다.
        print("엔진(127.0.0.1:7860)에 생성 요청 전송 중...")
        response = requests.post(url=f'{WEBUI_URL}/sdapi/v1/img2img', json=payload)
        response.raise_for_status() # 오류가 있으면 예외 발생

        r = response.json()

        # 3. 응답으로 받은 Base64 이미지를 디코딩해서 PIL 이미지로 변환합니다.
        if 'images' in r and len(r['images']) > 0:
            image_data = base64.b64decode(r['images'][0])
            result_image = Image.open(io.BytesIO(image_data))
            
            print("엔진으로부터 이미지 수신 완료. 프론트엔드로 반환합니다.")
            
            # 4. 완성된 이미지를 프론트엔드(Gradio UI)로 반환합니다.
            return result_image
        else:
            print("API 응답에 이미지가 없습니다:", r)
            return None # 오류 발생 시 아무것도 반환하지 않음

    except requests.exceptions.RequestException as e:
        print(f"API 요청 중 치명적 오류 발생: {e}")
        print("WebUI(엔진)가 --api 옵션과 함께 켜져 있는지 확인하세요.")
        return None

# --- 프론트엔드 UI (팀원 작업 공간) ---
# 임시용

with gr.Blocks() as demo:
    gr.Markdown("# 🤖 AI 스프라이트 시트 생성기 (매니저 API)")
    with gr.Row():
        # 입력값
        char_img = gr.Image(type="pil", label="캐릭터 이미지 업로드")
        motion = gr.Dropdown(choices=["걷기", "달리기", "점프"], label="모션 선택")
        # 출력값
        output_img = gr.Image(label="결과 이미지 (테스트)")
    
    btn = gr.Button("생성하기!")
    
    # '생성하기' 버튼을 누르면 generate_sprite 함수가 실행됩니다.
    btn.click(fn=generate_sprite, inputs=[char_img, motion], outputs=[output_img])

# --- '매니저 API' 서버 실행 ---

# 0.0.0.0: 모든 IP에서의 접속을 허용 (Local + Public)
# server_port=8000: '매니저'가 사용할 포트 번호
# share=True: 팀원이 접속할 수 있는 공개 주소 생성
print("매니저 API 서버(127.0.0.1:8000)를 시작합니다...")
demo.launch(server_name="0.0.0.0", server_port=8000, share=True)