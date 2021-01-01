import sys
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

    # 構造Marker圖像
    marker = np.zeros_like(img_binary)
    marker[0, :] = 255
    marker[-1, :] = 255
    marker[:, 0] = 255
    marker[:, -1] = 255

    # 原圖取補得到MASK圖像,限制膨脹結果
    mask = 255 - img_binary
    cv2.imshow("mask", mask)  # mask

    # Region Filling，定義一個Structures Element 做dilation，直到收斂為止
    el = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    cnt = 0
    m_cnt = 0
    while True:
        marker_pre = marker
        dilation = cv2.dilate(marker, el)
        marker = np.min((dilation, mask), axis=0)  # dilation 跟 mask做比較，取最小值(0)，以達到filling的效果
        # 觀察 marker 變化
        # cnt += 1
        # if cnt > 20:
        #     cv2.imshow(f"marker{m_cnt}", marker)
        #     m_cnt += 1
        #     cnt = 0
        if (marker_pre == marker).all():
            break

    dst = 255 - marker
    cv2.imshow("dst", dst)  # dst

    filling = dst - img_binary
    cv2.imshow("filling", filling)  # filling

    # 隨意Key一鍵結束程式
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
