import gradio as gr
import requests
import base64
from PIL import Image
import io
import numpy as np
import cv2
import os
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------
# 1. 환경 설정 및 초기화
# ---------------------------------------------------------
load_dotenv()

# OpenAI 클라이언트 (API Key가 .env 파일에 있어야 함)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Stable Diffusion WebUI 서버 주소 (서버 켤 때마다 확인 필요)
WEBUI_URL = ""
CONTROLNET_MODEL_NAME = "kohya_controllllite_xl_canny [2ed264be]"

# ---------------------------------------------------------
# 2. 유틸리티 함수
# ---------------------------------------------------------
def pil_to_base64(pil_image):
    with io.BytesIO() as stream:
        pil_image.save(stream, "PNG", pnginfo=None)
        return base64.b64encode(stream.getvalue()).decode('utf-8')

def process_pony_prompt(user_prompt, negative_prompt):
    # Pony 모델용 퀄리티 태그 자동 주입
    quality_tags = "score_9, score_8_up, score_7_up, score_6_up, source_anime, high quality, "
    full_prompt = quality_tags + user_prompt
    
    base_negative = "score_4, score_5, score_6, low quality, bad anatomy, worst quality, text, watermark, "
    full_negative = base_negative + negative_prompt
    return full_prompt, full_negative

# ---------------------------------------------------------
# 3. 챗봇 로직 (수동 Chatbot 대응)
# ---------------------------------------------------------
def chat_response(message, history):
    """
    사용자의 메시지를 받아 GPT 답변을 생성하고,
    Gradio 4.x 표준 포맷인 [딕셔너리 리스트]를 반환합니다.
    """
    # ★ 수정됨: 사용자가 '제외'를 요청하면 Negative Prompt에 추가하도록 지능형 지시 ★
    system_prompt = (
        "너는 Stable Diffusion(Pony XL) 전문가인 AI 아트 어시스턴트야. "
        "사용자가 일상적인 대화를 하면 한국어로 친절하게 답해줘. "
        "하지만 사용자가 '그림 그려줘', '프롬프트 짜줘' 같은 요청을 하면, "
        "반드시 'Danbooru 스타일의 태그(Tag)' 형식으로 영어 프롬프트를 작성해야 해. "
        "문장이 아니라 쉼표(,)로 구분된 단어들을 나열해줘. "
        "\n"
        "★중요★: 사용자가 '투구는 빼줘', '안경 없이' 같이 특정 요소를 제외해달라고 하면, "
        "그 단어(예: helmet, glasses)를 반드시 'Negative Prompt' 맨 앞에 추가해줘."
        "\n"
        "답변은 반드시 아래 형식을 지켜줘:\n"
        "설명: (한국어로 그림에 대한 설명)\n"
        "Positive Prompt: (복사해서 쓸 수 있는 영문 태그 나열. 예: 1girl, solo, red armor...)\n"
        "Negative Prompt: (제외할 단어들 + 기본 부정 태그들. 예: helmet, beard, low quality, bad anatomy, extra fingers, mutation...)"
    )
    
    # history는 이제 [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}] 형태입니다.
    
    # 1. GPT에게 보낼 메시지 구성 (시스템 프롬프트 + 이전 대화)
    messages = [{"role": "system", "content": system_prompt}]
    
    for msg in history:
        messages.append({"role": msg['role'], "content": msg['content']})
    
    # 현재 질문 추가
    messages.append({"role": "user", "content": message})

    # 2. OpenAI API 호출
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=600
        )
        bot_reply = response.choices[0].message.content
    except Exception as e:
        bot_reply = f"오류가 발생했습니다: {str(e)}"
    
    # 3. Gradio 화면 업데이트 (딕셔너리 형태로 추가)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": bot_reply})
    
    return "", history # 입력창 비우기, 히스토리 업데이트

# ---------------------------------------------------------
# 4. 이미지 생성 로직
# ---------------------------------------------------------
def generate_canny(sketch_dict, prompt_text, negative_prompt):
    if sketch_dict is None:
        return None, None # 예외 처리

    # 배경(선화용)과 채색 레이어 분리
    clean_line_art = sketch_dict["background"]
    colored_draft = sketch_dict["composite"]
    
    if not clean_line_art:
        clean_line_art = colored_draft

    # 리사이징
    width, height = 1024, 1024
    clean_resized = clean_line_art.resize((width, height))
    colored_resized = colored_draft.resize((width, height))
    
    # Canny 추출
    image_np = np.array(clean_resized)
    if image_np.shape[-1] == 4: image_np = image_np[:, :, :3]
    canny_np = cv2.Canny(image_np, 50, 100) 
    canny_image_pil = Image.fromarray(canny_np)

    init_base64 = pil_to_base64(colored_resized)
    canny_base64 = pil_to_base64(canny_image_pil)
    final_prompt, final_negative = process_pony_prompt(prompt_text, negative_prompt)

    payload = {
        "prompt": final_prompt,
        "negative_prompt": final_negative,
        "init_images": [init_base64], 
        "steps": 28,
        "width": width,
        "height": height,
        "cfg_scale": 7.0,
        "sampler_name": "Euler a",
        "denoising_strength": 0.85, 
        "alwayson_scripts": {
            "controlnet": {
                "args": [{
                    "image": canny_base64,
                    "module": "none", 
                    "model": CONTROLNET_MODEL_NAME,
                    "weight": 1.2,
                    "control_mode": "ControlNet is more important",
                }]
            }
        }
    }

    try:
        response = requests.post(url=f'{WEBUI_URL}/sdapi/v1/img2img', json=payload, timeout=600)
        response.raise_for_status()
        r = response.json()
        if 'images' in r:
            return Image.open(io.BytesIO(base64.b64decode(r['images'][0]))), canny_image_pil
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def generate_inpaint(image_editor_dict, prompt_text, negative_prompt):
    if not image_editor_dict or not image_editor_dict["layers"]:
        return None

    init_img = image_editor_dict["background"]
    mask_layer = image_editor_dict["layers"][0]

    mask_np = np.array(mask_layer)
    if mask_np.shape[2] == 4:
        mask_image = Image.fromarray(mask_np[:, :, 3]).convert("L")
    else:
        mask_image = mask_layer.convert("L")

    width, height = 1024, 1024
    init_img_resized = init_img.resize((width, height))
    mask_img_resized = mask_image.resize((width, height))

    init_base64 = pil_to_base64(init_img_resized)
    mask_base64 = pil_to_base64(mask_img_resized)
    final_prompt, final_negative = process_pony_prompt(prompt_text, negative_prompt)

    payload = {
        "prompt": final_prompt,
        "negative_prompt": final_negative,
        "init_images": [init_base64],
        "mask": mask_base64,
        "steps": 35,
        "width": width,
        "height": height,
        "cfg_scale": 7.0,
        "sampler_name": "Euler a",
        "mask_blur": 4,
        "inpainting_fill": 1,
        "inpaint_full_res": True,
        "denoising_strength": 0.75
    }

    try:
        response = requests.post(url=f'{WEBUI_URL}/sdapi/v1/img2img', json=payload, timeout=600)
        response.raise_for_status()
        r = response.json()
        if 'images' in r:
            return Image.open(io.BytesIO(base64.b64decode(r['images'][0])))
    except Exception as e:
        print(f"Error: {e}")
        return None

# ---------------------------------------------------------
# 5. 통합 UI 구성
# ---------------------------------------------------------
with gr.Blocks() as demo:
    gr.HTML("""<style>footer {visibility: hidden; display: none !important;}</style>""")

    gr.Markdown("# 🎨 AI Creative Studio")
    gr.Markdown("챗봇과 상의하여 프롬프트를 만들고, 아트 스튜디오에서 나만의 작품을 완성하세요.")

    with gr.Tabs():
        # [Tab 1] 챗봇 인터페이스
        with gr.TabItem("🤖 AI 프롬프트 챗봇"):
            gr.Markdown("### 💡 무엇을 그리고 싶으신가요?")
            
            chatbot = gr.Chatbot(label="대화창", height=400)
            msg = gr.Textbox(label="메시지 입력", placeholder="예: 투구 없는 기사 그려줘")
            clear = gr.Button("대화 지우기")

            # 이벤트 연결
            msg.submit(chat_response, [msg, chatbot], [msg, chatbot])
            clear.click(lambda: [], None, chatbot, queue=False)

        # [Tab 2] 아트 스튜디오
        with gr.TabItem("🎨 아트 스튜디오"):
            gr.Markdown("### 🛠️ 이미지를 업로드하고 수정하세요")
            
            with gr.Tabs():
                # [Sub-Tab A] 스케치 -> 이미지
                with gr.TabItem("스케치 완성"):
                    with gr.Row():
                        with gr.Column():
                            sketch_input = gr.ImageEditor(
                                label="스케치", 
                                type="pil", 
                                height=500,
                                brush=gr.Brush(colors=["#000000", "#FF0000", "#00FF00", "#0000FF"], default_size=4)
                            )
                            s_prompt = gr.Textbox(label="프롬프트 (Positive Prompt 복사)", placeholder="1girl, silver armor...")
                            s_neg = gr.Textbox(label="부정 프롬프트 (Negative Prompt 복사)", value="low quality, bad anatomy, worst quality, text, watermark")
                            s_btn = gr.Button("✨ 스케치로 생성하기", variant="primary")
                        
                        with gr.Column():
                            s_result = gr.Image(label="완성된 이미지")
                            s_debug = gr.Image(label="Canny 미리보기", height=200)

                    s_btn.click(generate_canny, [sketch_input, s_prompt, s_neg], [s_result, s_debug])

                # [Sub-Tab B] 인페인팅
                with gr.TabItem("부분 수정"):
                    with gr.Row():
                        with gr.Column():
                            inpaint_input = gr.ImageEditor(
                                label="수정할 이미지", 
                                type="pil", 
                                height=500,
                                brush=gr.Brush(colors=["#FFFFFF"], default_size=15)
                            )
                            i_prompt = gr.Textbox(label="수정 내용", placeholder="red eyes...")
                            i_neg = gr.Textbox(label="부정 프롬프트", value="low quality, bad anatomy")
                            i_btn = gr.Button("🖌️ 부분 수정하기", variant="primary")
                        
                        with gr.Column():
                            i_result = gr.Image(label="수정된 이미지")

                    i_btn.click(generate_inpaint, [inpaint_input, i_prompt, i_neg], [i_result])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)