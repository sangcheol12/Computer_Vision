import cv2 as cv
import numpy as np
import sys
import hw1_3

def hough_circles(file_path):
    black_circle = 0
    white_circle = 0
    median_bright = 0
    src = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
    if src is None:
        print("이미지를 불러올 수 없습니다.")
        exit()
    
    padding_size = 7
    src = cv.copyMakeBorder(src, padding_size, padding_size, padding_size, padding_size, cv.BORDER_CONSTANT, value=(255, 255, 255))
    
    src = hw1_3.auto_perspecive(src, 51, 176, 90, 79, 13, 1.0)
    # 이미지 크기 조정
    src = cv.resize(src, (600, 600))
    
    noise = np.zeros(src.shape, np.int32)
    cv.randn(noise, 0,5)
    cv.add(src, noise, src, dtype=cv.CV_8UC1)
    
    # Bilateral 필터로 잡음 제거
    blurred = cv.bilateralFilter(src, -1 , 10, 5)
    
    # 중앙값 필터로 추가적인 잡음 제거
    median_filtered = cv.medianBlur(blurred, 5)  # 5x5 크기의 필터 사용 (크기 조절 가능)
    
    circles = cv.HoughCircles(median_filtered, cv.HOUGH_GRADIENT, 1, 50, param1=140, param2=12, minRadius= 0, maxRadius= 30)
    
    dst = cv.cvtColor(src, cv.COLOR_GRAY2BGR)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))  # 소수점 좌표를 정수로 변환
        for i in circles[0, :]:
            cx, cy, radius = i
            cv.circle(dst, (cx, cy), radius, (0, 0, 255), 2, cv.LINE_AA)
            median_bright += median_filtered[cy,cx]
        median_bright /= circles.shape[1]

    if circles is None:
        print("원이 존재하지 않습니다.")
        return
    
    for i in range(circles.shape[1]):
        cx, cy, radius = circles[0][i]

        if (median_filtered[cy ,cx] / median_bright) >= 1:
            white_circle += 1
        else:
            black_circle += 1
    
    print('w:' + str(white_circle) + ' b:' + str(black_circle))
    
    cv.imshow('Circles', dst)
    cv.waitKey()
    cv.destroyAllWindows()
    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python hw1_4.py <이미지 파일 경로>")
    else:
        file_path = sys.argv[1]
        hough_circles(file_path)
