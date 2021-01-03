import sys
import cv2


def main():
    img_path = sys.argv[1]
    element_size = int(sys.argv[2])
    input_img = cv2.imread(img_path)
    cv2.imshow("source", input_img)

    gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    # 二值化
    thresh = 128
    img_binary = cv2.threshold(gray_img, thresh, 255, cv2.THRESH_BINARY)[1]

    # 3*3 Structures Element
    origin_el = cv2.getStructuringElement(cv2.MORPH_RECT, (element_size, element_size))
    # Erosion
    erode = cv2.erode(img_binary, origin_el)
    # Dilation
    dilate = cv2.dilate(img_binary, origin_el)

    # 膨脹 - 腐蝕 = 邊界 (也可膨脹-原圖 或者 原圖-腐蝕，高興就好!!)
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
    if len(sys.argv) != 3:
        print("缺少要處理的圖片名稱")
        exit()

    main()
