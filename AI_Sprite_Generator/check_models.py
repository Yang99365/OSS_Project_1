import requests
import json

# AI 엔진(A1111 WebUI) 주소
WEBUI_URL = "https://5161fef20484.ngrok-free.app"

try:
    print(f"엔진({WEBUI_URL})에게 ControlNet 모델 목록을 요청합니다...")
    response = requests.get(url=f'{WEBUI_URL}/controlnet/model_list')
    response.raise_for_status()

    models = response.json()

    print("\n--- [성공] 엔진이 응답한 ControlNet 모델 목록 ---\n")
    print(json.dumps(models, indent=2))
    print("\n-------------------------------------------------")
    print("\n[다음 할 일: app.py의 ControlNet 모델명 수정]")
    print("이 목록에서 'canny'가 포함된 정확한 문자열을 복사하세요.")
    print("app.py의 두 군데(Canny 생성, Inpaint 생성) 'model' 값에 붙여넣으세요.")

except requests.exceptions.RequestException as e:
    print(f"\n[오류] 엔진에 연결할 수 없습니다: {e}")
    print("A1111 WebUI (webui-user.bat)가 '--api' 옵션과 함께 켜져 있는지 확인하세요.")