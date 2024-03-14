import cv2 as cv
import numpy as np
import math
import sys

def load_image(image_path):
    # 이미지 읽기
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if image is None:
        print('이미지를 불러올 수 없습니다.')
        exit()

    # 이미지 주변에 패딩 추가
    padding_size = 7
    image = cv.copyMakeBorder(image, padding_size, padding_size, padding_size, padding_size, cv.BORDER_CONSTANT, value=(255, 255, 255))
    cv.imshow('src', image)

    return image

def apply_contrast(image, alpha):
    return cv.convertScaleAbs(image, alpha=1 + alpha, beta=-70 * alpha)

def detect_edges(src, canny_low, canny_high):
    return cv.Canny(src, canny_low, canny_high)

def classify_four_points(points):
    top_left = min(points, key=lambda point: point[0] + point[1])
    top_right = max(points, key=lambda point: point[0] - point[1])
    bottom_left = min(points, key=lambda point: point[0] - point[1])
    bottom_right = max(points, key=lambda point: point[0] + point[1])
    return top_left, top_right, bottom_right, bottom_left

def hough_transform(src, hough_threshold, hough_min_len, hough_min_gap):
    lines = cv.HoughLinesP(src, 1, math.pi / 180, hough_threshold, minLineLength=hough_min_len, maxLineGap=hough_min_gap)
    return lines

def warp_perspective(src, src_pts, width, height):
    pers_mat = cv.getPerspectiveTransform(np.array(src_pts).astype(np.float32), np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]]).astype(np.float32))
    result = cv.warpPerspective(src, pers_mat, (width, height))
    return result

def auto_perspecive(src, canny_low, canny_high, hough_threshold, hough_min_len, hough_min_gap, alpha):
    # 화소 처리
    contrast = apply_contrast(src, alpha)
    
    # 꼭직점 검출
    edge = detect_edges(contrast, canny_low, canny_high)
    
    # Hough 변환
    lines = hough_transform(edge, hough_threshold, hough_min_len, hough_min_gap)
    
    dst = cv.cvtColor(edge, cv.COLOR_GRAY2BGR)
    
    points = [] 
    if lines is not None:
        for i in range(lines.shape[0]):
            pt1 = (lines[i][0][0], lines[i][0][1])
            pt2 = (lines[i][0][2], lines[i][0][3])
            points.append(pt1)
            points.append(pt2)

    top_left, top_right, bottom_right, bottom_left = classify_four_points(points)

    src_pts = [top_left, top_right, bottom_right, bottom_left]
    
    # 투시 변환
    w, h = 600, 600
    result = warp_perspective(src, src_pts, w, h)

    return result

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python hw1_2.py <이미지 파일 경로>")
    else:
        file_path = sys.argv[1]
        image = load_image(file_path)
        result = auto_perspecive(image, 51, 176, 90, 79, 13, 1.0)
        cv.imshow('dst', result)

        cv.waitKey()
        cv.destroyAllWindows()
