import sys
from math import sqrt
import cv2
import numpy as np

if len(sys.argv) != 2:
    print("缺少要處理的圖片名稱")
    exit()

img_path = sys.argv[1]


def main():
    input_img = cv2.imread(img_path)
    cv2.imshow("source", input_img)

    gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    # 二值化
    thresh = 128
    img_binary = cv2.threshold(gray_img, thresh, 255, cv2.THRESH_BINARY)[1]

    # 3*3 Structures Element
    origin_el = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # Erosion
    erode = cv2.erode(img_binary, origin_el)
    # Dilation
    dilate = cv2.dilate(img_binary, origin_el)

    # 膨脹 - 腐蝕 = 邊界
    boundary = cv2.absdiff(dilate, erode)

    # 二值圖畫素取反
    result = cv2.bitwise_not(boundary)

    cv2.imshow("erode", erode)  # erode
    cv2.imshow("dilate", dilate)  # dilate
    cv2.imshow("boundary", result)  # boundary

    # 隨意Key一鍵結束程式
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
