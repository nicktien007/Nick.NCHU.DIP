import sys
from math import sqrt
import cv2
import numpy as np

if len(sys.argv) != 2:
    print("缺少要處理的圖片名稱")
    exit()

img_path = sys.argv[1]
threshold = 128

mask = np.array([
    # Mean mask
    [1, 1, 1,
     1, 1, 1,
     1, 1, 1],

    # Sobel y mask
    [-1, -2, -1,
     0, 0, 0,
     1, 2, 1],

    # Sobel x mask
    [-1, 0, 1,
     -2, 0, 2,
     -1, 0, 1],

    # Laplace Mask
    [-1, -1, -1,
     -1, 8, -1,
     -1, -1, -1]])


def main():
    input_img = cv2.imread(img_path)
    cv2.imshow("Unsharp_source", input_img)

    gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    rows = gray_img.shape[0]
    cols = gray_img.shape[1]

    diff_1_img = np.zeros((rows, cols), dtype=gray_img.dtype)
    diff_2_img = np.zeros((rows, cols), dtype=gray_img.dtype)
    mean_img = np.zeros((rows, cols), dtype=gray_img.dtype)

    convolution(gray_img, diff_1_img, 1)  # sobel y
    convolution(gray_img, diff_2_img, 2)  # sobel x

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # print(f"a,b={diff_1_img[i, j]},{diff_2_img[i, j]}")
            sobel = int(diff_1_img[i, j]) + int(diff_2_img[i, j])
            # if sobel > 255:
            #     print(sobel)
            # print(f"i,j={i},{j}")
            diff_1_img[i, j] = chk_val(sobel)

    convolution(gray_img, diff_2_img, 3)  # laplac
    convolution(diff_1_img, mean_img, 0)

    for i in range(1, diff_2_img.shape[0] - 1):
        for j in range(diff_2_img.shape[1] - 1):
            for c in range(3):
                input_img[i, j, c] = chk_val(input_img[i, j, c] + (diff_2_img[i, j] * (mean_img[i, j] / 255)))

    cv2.imshow("Unsharp_diff_1_img", diff_1_img)  # sobel
    cv2.imshow("Unsharp_mean_img", mean_img)  # mean sobel
    cv2.imshow("Unsharp_diff_2_img", diff_2_img)  # laplac
    cv2.imshow("Unsharp_UnsharpMasking", input_img)
    # 隨意Key一鍵結束程式
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def convolution(src, out, type):
    rows = out.shape[0]
    cols = out.shape[1]

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            conv = mask[type][0] * src[i - 1, j - 1] + \
                   mask[type][1] * src[i - 1, j] + \
                   mask[type][2] * src[i - 1, j + 1] + \
                   mask[type][3] * src[i, j - 1] + \
                   mask[type][4] * src[i, j] + \
                   mask[type][5] * src[i, j + 1] + \
                   mask[type][6] * src[i + 1, j - 1] + \
                   mask[type][7] * src[i + 1, j] + \
                   mask[type][8] * src[i + 1, j + 1]

            if type == 0:
                conv = conv / 9
                conv = 255 if conv > threshold else 0
            elif type == 1 or type == 2:
                conv = sqrt(pow(conv, 2))

            out[i, j] = chk_val(conv)


def chk_val(v):
    if v > 255:
        v = 255
    elif v < 0:
        v = 0
    return v


if __name__ == '__main__':
    main()
