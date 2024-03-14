import sys

t = int(sys.stdin.readline())

for _ in range(t):
    sum_val = 1
    n, m = map(int, sys.stdin.readline().split())
    dq = list(map(int, sys.stdin.readline().split()))
    for x in dq:
        sum_val *= x
    
    str_input = sys.stdin.readline().strip()
    res = []
    for i in str_input:
        res.append(str(sum_val % m))
        if i == 'L':
            x = dq.pop(0)
            sum_val //= x
        else:
            x = dq.pop()
            sum_val //= x
    sys.stdout.write(' '.join(res) + '\n')




'''import cv2 as cv
import numpy as np
import sys

pts_cnt = 0
pts = np.zeros((4, 2), dtype=np.float32)
draw = 0
rows, cols = 0,0

def onMouse(event, x, y, flags, param):
    global pts_cnt

    if event == cv.EVENT_LBUTTONDOWN:
        # 좌표에 동그라미 표시
        cv.circle(draw, (x, y), 3, (255, 0, 0), -1)
        cv.imshow('src', draw)

        pts[pts_cnt] = [int(x), int(y)]
        pts_cnt += 1

        if pts_cnt == 4:
            # 좌표 4개 중 상하좌우 찾기
            sm = pts.sum(axis=1)  # 4쌍의 좌표 각각 x+y 계산
            diff = np.diff(pts, axis=1)  # 4쌍의 좌표 각각 x-y 계산

            topLeft = pts[np.argmin(sm)]  # x+y가 가장 작은 값이 좌상단 좌표
            topRight = pts[np.argmin(diff)]  # x-y가 가장 작은 것이 우상단 좌표
            bottomRight = pts[np.argmax(sm)]  # x+y가 가장 큰 값이 우하단 좌표
            bottomLeft = pts[np.argmax(diff)]  # x-y가 가장 큰 값이 좌하단 좌표

            # 변환 전 4개 좌표 
            pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])

            # 변환 후 영상에 사용할 서류의 폭과 높이 계산
            w1 = abs(bottomRight[0] - bottomLeft[0])
            w2 = abs(topRight[0] - topLeft[0])
            h1 = abs(topRight[1] - bottomRight[1])
            h2 = abs(topLeft[1] - bottomLeft[1])
            width = int(max([w1, w2]))  # 두 좌우 거리간의 최대값이 서류의 폭
            height = int(max([h1, h2]))  # 두 상하 거리간의 최대값이 서류의 높이

            # 변환 후 4개 좌표
            pts2 = np.float32([[0, 0], [width - 1, 0],
                               [width - 1, height - 1], [0, height - 1]])

            # 변환 행렬 계산 
            mtrx = cv.getPerspectiveTransform(pts1, pts2)
            # 투시 변환 적용
            result = cv.warpPerspective(param, mtrx, (width, height))
            cv.imshow('dst', result)

def detect_checkerboard(file_path):
    global draw
    global rows,cols
    
    image = cv.imread(file_path)

    if image is None:
        print("이미지를 불러올 수 없습니다.")
        exit()
    
    image = cv.resize(image, (600, 600))
        
    rows, cols = image.shape[:2]
    draw = image.copy()
        
    cv.imshow('src', image)
    cv.setMouseCallback('src', onMouse, param=image)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python hw1_2.py <이미지 파일 경로>")
    else:
        file_path = sys.argv[1]
        detect_checkerboard(file_path)'''
