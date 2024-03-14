import cv2 as cv
import torch
import sys
from pathlib import Path

# YOLOv5 모델 로드
def find_empire(imgPath):
    res = False
    model_path = Path("./best.pt")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt', force_reload=True)

    img = cv.imread(imgPath)

    # YOLOv5를 통한 객체 탐지
    yoloImg = model(img)

    # 탐지된 객체 정보 가져오기
    for det in yoloImg.xyxy[0]:
        x, y, w, h, conf, cls = map(float, det)

        # 객체 존재 유무 확인
        if conf > 0.7:
            res = True
    print(res)
            


if __name__ == "__main__":
    find_empire(sys.argv[1])