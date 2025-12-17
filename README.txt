[오픈소스SW 팀프로젝트 제출]
프로젝트명: AI Creative Studio

[시스템 구조 설명]

Client (제출한 코드): Gradio 기반의 웹 인터페이스 및 전처리 로직 (CPU 환경 구동 가능)

Server (외부 연동): 고성능 GPU가 필요한 Stable Diffusion WebUI는 별도 서버(API)와 통신합니다.


[파일 구성]

art_assistant.py : 메인 실행 코드 (Gradio 웹 애플리케이션)

requirements.txt : 프로젝트 실행에 필요한 라이브러리 목록

.env : OpenAI API Key 설정 파일

[실행 방법 (Client)]

파이썬(Python 3.10 이상)이 설치되어 있어야 합니다.

터미널에서 다음 명령어로 필수 라이브러리를 설치해주세요.
pip install -r requirements.txt

다음 명령어로 앱을 실행합니다.
python art_assistant.py

터미널에 표시되는 Gradio 제공 링크로 접속합니다.

[주의 사항]

실행 시 .env 파일에 유효한 OPENAI_API_KEY가 들어있는지 확인해주세요.

art_assistant.py 코드 내의 WEBUI_URL 변수에 유효한 Stable Diffusion 서버 주소가 설정되어 있어야 이미지 생성이 가능합니다.

[부록: Stable Diffusion 서버 직접 구축 가이드 (Stability Matrix 활용)]

스테이블 디퓨전 설치 for art_assistant.doc 참고