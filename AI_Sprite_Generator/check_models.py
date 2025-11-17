import requests
import json

# AI 엔진(A1111 WebUI) 주소
WEBUI_URL = "http://127.0.0.1:7860"

try:
    print(f"엔진({WEBUI_URL})에게 ControlNet 모델 목록을 요청합니다...")
    response = requests.get(url=f'{WEBUI_URL}/controlnet/model_list')
    response.raise_for_status()

    models = response.json()

    print("\n--- [성공] 엔진이 응답한 ControlNet 모델 목록 ---")
    print(json.dumps(models, indent=2))
    print("-------------------------------------------------")
    print("\n[다음 할 일]")
    print("이 목록('model_list')에 있는 이름 중, 'canny'가 포함된")
    print("정확한 문자열(예: 'control_v11p_sd15_canny [d14c016b]')을 복사해서")
    print("app.py의 'model' 값으로 붙여넣으세요.")

except requests.exceptions.RequestException as e:
    print(f"\n[오류] 엔진에 연결할 수 없습니다: {e}")
    print("A1111 WebUI (webui-user.bat)가 '--api' 옵션과 함께 켜져 있는지 확인하세요.")