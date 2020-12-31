import sys
from math import fabs
import cv2
import numpy as np

if len(sys.argv) != 2:
    print("缺少要辨識的圖片名稱")
    exit()

# 需要辨識的人臉圖片名稱
img_path = sys.argv[1]
MAX_MATRIX = 8
A = np.zeros((MAX_MATRIX, MAX_MATRIX), dtype=np.float64)
# A = np.zeros((MAX_MATRIX, MAX_MATRIX))
# b = [270, 273, 690, 694, 69, 458, 164, 518]  # Set upMatrix
b = [14, 6, 752, 762, 101, 550, 28, 575]


# b = np.zeros(MAX_MATRIX, dtype=np.float64)  # Set upMatrix

class Corner:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def gauss_jordan():
    i, j, k, s = 0, 0, 0, 0
    r, u, temp = 0.0, 0.0, 0.0
    l = np.zeros((MAX_MATRIX, MAX_MATRIX), dtype=np.float64)

    for k in range(MAX_MATRIX - 1):
        s = k
        r = fabs(A[k, k])

        for i in range(k, MAX_MATRIX):
            if r < fabs(A[i, k]):
                r = fabs(A[i, k])
                s = i
        if s != k:
            A[[k, s]] = A[[s, k]]  # 交換 k row 和 s rom
            temp = b[k]
            b[k] = b[s]
            b[s] = temp

        for i in range(k + 1, MAX_MATRIX):
            l[i, k] = A[i, k] / A[k, k]

            for j in range(k, MAX_MATRIX):
                A[i, j] = A[i, j] - l[i, k] * A[k, j]
            b[i] = b[i] - l[i, k] * b[k]

    if A[k, k] < 0.0001:
        print("Error!Can not find the solution!")
        exit(1)

    for i in range(MAX_MATRIX - 1, -1, -1):
        u = 0
        for j in range(i + 1, MAX_MATRIX):
            u = u + A[i, j] * b[j]
        b[i] = (b[i] - u) / A[i, i]


def main():
    image = cv2.imread(img_path)

    # aa=cv2.cv.CreateMat(500, 500, image.dtype)
    # persImg = np.zeros(image.shape, dtype=np.float64)

    # persImg = np.zeros((400, 480, 3), dtype=image.dtype)
    # persImg = np.zeros((520, 700, 3), dtype=image.dtype)
    persImg = np.zeros((600, 800, 3), dtype=image.dtype)

    corner = []
    rows = persImg.shape[0]
    cols = persImg.shape[1]

    corner.append(Corner(0, 0))
    corner.append(Corner(rows - 1, 0))
    corner.append(Corner(0, cols - 1))
    corner.append(Corner(rows - 1, cols - 1))

    for i in range(MAX_MATRIX):
        if i < 4:
            A[i, 0] = corner[i].x
            A[i, 1] = corner[i].y
            A[i, 2] = corner[i].x * corner[i].y
            A[i, 3] = 1
        else:
            A[i, 4] = corner[i - 4].x
            A[i, 5] = corner[i - 4].y
            A[i, 6] = corner[i - 4].x * corner[i - 4].y
            A[i, 7] = 1

    # b[0] = 270 #x1
    # b[1] = 273 #x2
    # b[2] = 690 #x3
    # b[3] = 694 #x4
    # b[4] = 69  #y1
    # b[5] = 458 #y2
    # b[6] = 164 #y3
    # b[7] = 518 #y4

    gauss_jordan()

    for i in range(MAX_MATRIX):
        print(f'b[{i}]={b[i]}')

    for i in range(rows):
        for j in range(cols):
            double_y = ((b[0] * i) + (b[1] * j) + (b[2] * i * j) + b[3])
            double_x = ((b[4] * i) + (b[5] * j) + (b[6] * i * j) + b[7])
            y = int(double_y)
            x = int(double_x)
            v = (double_y - y)
            u = (double_x - x)
            # print(f"i,j={i},{j}")
            # print(f"x,y={x},{y}")
            for c in range(3):
                # r1 = (1 - u) * (1 - v) * image[x, y][c]
                # u1 = u * (1 - v) * image[x + 1, y][c]
                # v1 = v * (1 - u) * image[x, y + 1][c]
                # u2 = u * v * image[x + 1, y + 1][c]
                #
                # rr = r1 + u1 + v1 + u2
                # print(f"rr={rr},r1={r1},u1={u1},v1={v1},u2={u2}")

                # persImg[i, j][c] = rr
                # rr = (1 - u) * (1 - v) * persImg[x, y][c] + \
                #      u * (1 - v) * persImg[x + 1, y][c] + \
                #      v * (1 - u) * persImg[x, y + 1][c] +\
                #      u * v * persImg[x + 1, y + 1][c]
                rr = (1 - u) * (1 - v) * image[x, y][c] + \
                     u * (1 - v) * image[x, y + 1][c] + \
                     v * (1 - u) * image[x + 1, y][c] + \
                     u * v * image[x + 1, y + 1][c]
                # print(f"rr={rr}")
                # persImg[x, y][c] = rr
                persImg[i, j][c] = rr

    cv2.imshow("Perspective_source", image)
    cv2.imshow("Perspective_process", persImg)
    # 隨意Key一鍵結束程式
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
