import sys
from math import fabs
import cv2
import numpy as np


class Corner:
    def __init__(self, r, c):
        self.r = r
        self.c = c

    def get_corner(self):
        return [Point(0, 0),
                Point(self.r - 1, 0),
                Point(0, self.c - 1),
                Point(self.r - 1, self.c - 1)]


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def main():
    img_path = sys.argv[1]
    input_img = cv2.imread(img_path)
    max_matrix = 8

    # b = [404, 275, 3376, 3520, 1077, 1636, 1198, 1674]  #Perspective_2.jpg
    # b = [66, 54, 1092, 1100, 174, 904, 61, 948]  #Perspective_3.jpg
    # x1,x2,x3,x4,y1,y2,y3,y4
    b = [int(b) for b in sys.argv[2].split(',')]
    A = np.zeros((max_matrix, max_matrix), dtype=np.float64)

    out_x = abs(b[3] - b[0])
    out_y = abs(b[4] - b[5])
    output_img = np.zeros((out_y, out_x, 3), dtype=input_img.dtype)  # Perspective_3.jpg

    rows = output_img.shape[0]
    cols = output_img.shape[1]
    corner = Corner(rows, cols).get_corner()

    init_A(A, corner, max_matrix)
    b = gauss_jordan(A, b)

    [print(f'b[{i}]={b[i]}') for i in range(max_matrix)]

    invert_mapping(b, input_img, output_img)

    cv2.imshow("Perspective_source", input_img)
    cv2.imshow("Perspective_process", output_img)
    # 隨意Key一鍵結束程式
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def init_A(A, corner, max_matrix):
    for i in range(max_matrix):
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


def gauss_jordan(A, b):
    """
    高斯消去法
    :return: b
    """
    i, j, k, s = 0, 0, 0, 0

    # l = np.zeros((MAX_MATRIX, MAX_MATRIX), dtype=np.float64)
    max_matrix = A.shape[0]
    l = np.zeros((max_matrix, max_matrix), dtype=np.float64)

    for k in range(max_matrix - 1):
        s = k
        r = fabs(A[k, k])

        for i in range(k, max_matrix):
            if r < fabs(A[i, k]):
                r = fabs(A[i, k])
                s = i
        if s != k:
            A[[k, s]] = A[[s, k]]  # 交換 k row 和 s row
            b[k], b[s] = b[s], b[k]

        for i in range(k + 1, max_matrix):
            l[i, k] = A[i, k] / A[k, k]

            for j in range(k, max_matrix):
                A[i, j] = A[i, j] - l[i, k] * A[k, j]
            b[i] = b[i] - l[i, k] * b[k]

    if A[k, k] < 0.0001:
        print("Error!Can not find the solution!")
        exit(1)

    for i in range(max_matrix - 1, -1, -1):
        u = 0
        for j in range(i + 1, max_matrix):
            u = u + A[i, j] * b[j]
        b[i] = (b[i] - u) / A[i, i]

    return b


def invert_mapping(b, input_img, output_img):
    rows = output_img.shape[0]
    cols = output_img.shape[1]

    for i in range(rows):
        for j in range(cols):
            double_y = ((b[0] * i) + (b[1] * j) + (b[2] * i * j) + b[3])
            double_x = ((b[4] * i) + (b[5] * j) + (b[6] * i * j) + b[7])
            y = int(double_y)
            x = int(double_x)
            v = (double_y - y)
            u = (double_x - x)
            for c in range(3):
                # f(x,y) = (1-u)(1-v)g(x,y) + u(1-v)g(x,y+1) + v(1-u)g(x+1,y) + uvg(x+1,y+1)
                output_img[i, j, c] = (1 - u) * (1 - v) * input_img[x, y][c] + \
                                      u * (1 - v) * input_img[x, y + 1][c] + \
                                      v * (1 - u) * input_img[x + 1, y][c] + \
                                      u * v * input_img[x + 1, y + 1][c]


if __name__ == '__main__':
    main()
