import cv2 as cv
import torch
import sys
from pathlib import Path

# YOLOv5 모델 로드
def draw_box_empire(imgPath):
    model_path = Path("./best.pt")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt', force_reload=True)
    
    img = cv.imread(imgPath)

    # YOLOv5를 통한 객체 탐지
    yoloImg = model(img)

    # 탐지된 객체 정보 가져오기
    for det in yoloImg.xyxy[0]:
        x, y, w, h, conf, cls = map(float, det)

        # bounding box 그리기
        if conf > 0.6:
            cv.rectangle(img, (int(x), int(y)), (int(w), int(h)), (0, 0, 255), 2)
        print(cls)
    
    cv.imshow('res', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
            

if __name__ == "__main__":
    draw_box_empire(sys.argv[1])