import cv2 as cv
import numpy as np
import sys
import math

def hough_line_seg(file_path):
    src = cv.imread(file_path)
    
    if src is None:
        print("이미지를 불러올 수 없습니다.")
        exit()
    
    # 이미지 크기 조정
    src = cv.resize(src, (600, 600))
    
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    noise = np.zeros(src.shape, np.int32)
    cv.randn(noise, 0, 5)
    cv.add(src, noise, src, dtype=cv.CV_8UC1)
    
    # Bilateral 필터로 잡음 제거
    blurred = cv.bilateralFilter(src, -1, 10, 5)
    
    src = blurred
    
    # 이미지 전처리
    edge = cv.Canny(src, 100, 200)
    
    # Hough 변환 파라미터 조정
    lines = cv.HoughLines(edge, 1, math.pi / 180, 130)
    
    dst = cv.cvtColor(edge, cv.COLOR_GRAY2BGR)
    
    # 감지된 선들의 근접한 선을 하나로 병합하기 위한 변수
    merged_lines = []
    
    if lines is not None:
        # 선을 감지할 때, 선의 극좌표(rho, theta) 정보가 사용
        for i in range(lines.shape[0]):
            rho, theta = lines[i][0]
            
            # 선의 방향 코사인(cos)과 사인(sin)을 계산하여 두 선이 평행한지 확인
            a = np.cos(theta)
            b = np.sin(theta)
            
            # 감지된 두 점 계산
            x0 = a * rho
            y0 = b * rho
            
            # 이전에 병합된 선 중 가장 가까운 선을 탐색
            merge = False
            for merged_line in merged_lines:
                rho_m, theta_m = merged_line
                a_m = np.cos(theta_m)
                b_m = np.sin(theta_m)
                x0_m = a_m * rho_m
                y0_m = b_m * rho_m
                
                # 가까운 두 선의 거리를 계산
                distance = np.sqrt((x0 - x0_m) ** 2 + (y0 - y0_m) ** 2)
                
                # 만약 두 선이 30 픽셀 이내에 있다면, 이 두 선을 병합
                if distance < 30:
                    merge = True
                    break
            
            if not merge:
                merged_lines.append((rho, theta))
                
    horizontal_lines = 0
    vertical_lines = 0
    
    for rho, theta in merged_lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        
        angle = math.atan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
        
        if abs(angle) < math.pi / 4:
            horizontal_lines += 1
        else:
            vertical_lines += 1
            
        cv.line(dst, pt1, pt2, (0, 0, 255), 2, cv.LINE_AA)
    
    #print(f"가로 선의 수: {horizontal_lines}")
    #print(f"세로 선의 수: {vertical_lines}")
    if (horizontal_lines + vertical_lines) / 2 <= 10:
        print("8 x 8")
    else:
        print("10 x 10")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python hw1_1.py <이미지 파일 경로>")
    else:
        file_path = sys.argv[1]
        hough_line_seg(file_path)
