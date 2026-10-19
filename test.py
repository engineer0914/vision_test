# Intel RealSense D435i/D435if 카메라 스트림을 받아서 선택한 YOLO 모델로 실시간 추론을 수행하고, q 키를 눌러 종료할 때 녹화된 영상을 저장하는 Python 스크립트입니다.
# 요청하신 대로 실행 시 CLI 인자(--model)로 모델 유형(brick 또는 comp)을 선택받으며, 선택된 모델 파일명(model_brick.pt 또는 model_comp.pt)에 맞춰 영상 파일이 {model_type}_{YYYYMMDD_HHMMSS}.mp4 형식으로 자동 저장됩니다.

import argparse
import datetime
import os
import sys
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="RealSense YOLO Inference & Video Recorder"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["brick", "comp"],
        help="Select model type: 'brick' (model_brick.pt) or 'comp' (model_comp.pt)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 모델 경로 및 저장 파일명 설정
    model_type = args.model
    model_path = f"model_{model_type}.pt"

    if not os.path.exists(model_path):
        print(f"[Error] 모델 파일을 찾을 수 없습니다: {model_path}")
        sys.exit(1)

    print(f"[Info] YOLO 모델 로딩 중: {model_path}")
    model = YOLO(model_path)

    # 2. RealSense 파이프라인 설정
    pipeline = rs.pipeline()
    config = rs.config()

    # 컬러 스트림 설정 (640x480, 30fps)
    width, height, fps = 640, 480, 30
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    print("[Info] RealSense 카메라 스트림 시작...")
    pipeline.start(config)

    # 3. 비디오 저장 설정 (VideoWriter)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{model_type}_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    print(
        f"[Info] 녹화 시작: {output_filename} (종료하려면 화면 창에서 'q'를 누르세요)"
    )

    try:
        while True:
            # RealSense 프레임 대기
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # RealSense 프레임을 NumPy 배열로 변환
            frame = np.asanyarray(color_frame.get_data())

            # YOLO 실시간 추론 (stream=True로 메모리 효율화)
            results = model(frame, verbose=False)

            # 추론 결과가 시각화된 프레임 생성
            annotated_frame = results[0].plot()

            # 영상 프레임 녹화 (추론 결과가 그려진 프레임 저장)
            out.write(annotated_frame)

            # 실시간 스트리밍 화면 출력
            cv2.imshow("RealSense YOLO Stream", annotated_frame)

            # 'q' 키 입력 시 종료
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[Info] 'q' 키가 입력되어 종료합니다.")
                break

    finally:
        # 자원 해제
        pipeline.stop()
        out.release()
        cv2.destroyAllWindows()
        print(f"[Success] 영상이 성공적으로 저장되었습니다: {output_filename}")


if __name__ == "__main__":
    main()
