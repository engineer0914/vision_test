Intel RealSense D435i/D435if 카메라 스트림을 받아서 선택한 YOLO 모델로 실시간 추론을 수행하고, q 키를 눌러 종료할 때 녹화된 영상을 저장하는 Python 스크립트입니다.


실행 방법
터미널에서 원하시는 모델 종류를 인자로 전달하여 실행합니다.

brick 모델 사용 시:

```
python test.py --model brick
```
(저장 파일 예시: brick_20261019_220000.mp4)

comp 모델 사용 시:

```
python test.py --model comp
```
(저장 파일 예시: comp_20261019_220000.mp4)

코드 주요 특징
CLI Argument Parser: --model 인자를 통해 brick 또는 comp를 전달받아 model_brick.pt 또는 model_comp.pt를 자동으로 매핑합니다.

동적 파일명 생성: datetime 모듈을 활용하여 녹화 종료 시점의 날짜/시간을 포함한 파일명(comp_YYYYMMDD_HHMMSS.mp4)으로 저장됩니다.

RealSense native integration: pyrealsense2 라이브러리를 통해 D435if 카메라의 BGR 스트림을 직접 바인딩하여 안정적으로 프레임을 획득합니다.
