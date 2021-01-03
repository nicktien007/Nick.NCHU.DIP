import sys
from enum import Enum
from math import sqrt
import cv2
import numpy as np


def main():
    img_path = sys.argv[1]
    threshold = int(sys.argv[2])
    input_img = cv2.imread(img_path)
    cv2.imshow("Unsharp_source", input_img)

    gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    service = UnsharpService(threshold=threshold)
    diff_1_img = service.get_diff_1(gray_img)
    diff_2_img = service.get_diff_2(gray_img)
    mean_img = service.mean_filter(diff_1_img)  # unsharp mask

    enha_img = service.enhancement(input_img, diff_2_img, mean_img)

    cv2.imshow("Unsharp_diff_1_img", diff_1_img)  # sobel
    cv2.imshow("Unsharp_mean_img", mean_img)  # mean sobel
    cv2.imshow("Unsharp_diff_2_img", diff_2_img)  # laplac
    cv2.imshow("Unsharp_UnsharpMasking", enha_img)

    # 隨意Key一鍵結束程式
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class ConvolutionType(Enum):
    MeanFilter = 1
    Sobel_X = 2
    Sobel_Y = 3
    Laplace = 4


class UnsharpService:
    def __init__(self, threshold):
        self.__threshold = threshold

        self.__mean_mask = [1, 1, 1,
                            1, 1, 1,
                            1, 1, 1]

        self.__sobel_x_mask = [-1, 0, 1,
                               -2, 0, 2,
                               -1, 0, 1]

        self.__sobel_y_mask = [-1, -2, -1,
                               0, 0, 0,
                               1, 2, 1]

        self.__laplace_mask = [-1, -1, -1,
                               -1, 8, -1,
                               -1, -1, -1]

    def mean_filter(self, img):
        """
        算術平均濾波
        :param img:
        :return:
        """
        rows = img.shape[0]
        cols = img.shape[1]
        mean_img = np.zeros((rows, cols), dtype=img.dtype)
        self.__convolution(img, mean_img, self.__mean_mask, ConvolutionType.MeanFilter)
        return mean_img

    def __sobel_y(self, src, out):
        self.__convolution(src, out, self.__sobel_y_mask, ConvolutionType.Sobel_Y)

    def __sobel_x(self, src, out):
        self.__convolution(src, out, self.__sobel_x_mask, ConvolutionType.Sobel_X)

    def get_diff_1(self, img):
        """
        一階微分
        :param img:
        :return:
        """
        rows = img.shape[0]
        cols = img.shape[1]
        sobel_y_img = np.zeros((rows, cols), dtype=img.dtype)
        sobel_x_img = np.zeros((rows, cols), dtype=img.dtype)
        diff_1_img = np.zeros((rows, cols), dtype=img.dtype)

        self.__sobel_y(img, sobel_y_img)  # sobel y
        self.__sobel_x(img, sobel_x_img)  # sobel x

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                # print(f"a,b={diff_1_img[i, j]},{diff_2_img[i, j]}")
                sobel = int(sobel_x_img[i, j]) + int(sobel_y_img[i, j])
                # if sobel > 255:
                #     print(sobel)
                # print(f"i,j={i},{j}")
                diff_1_img[i, j] = self.__chk_val(sobel)
        return diff_1_img

    def get_diff_2(self, img):
        """
        二階微分
        :param img:
        :return:
        """
        rows = img.shape[0]
        cols = img.shape[1]

        diff_2_img = np.zeros((rows, cols), dtype=img.dtype)
        self.__laplace(img, diff_2_img)

        return diff_2_img

    def __laplace(self, src, out):
        self.__convolution(src, out, self.__laplace_mask, ConvolutionType.Laplace)

    def __convolution(self, src, out, mask, conv_type):
        rows = out.shape[0]
        cols = out.shape[1]

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                conv = mask[0] * src[i - 1, j - 1] + \
                       mask[1] * src[i - 1, j] + \
                       mask[2] * src[i - 1, j + 1] + \
                       mask[3] * src[i, j - 1] + \
                       mask[4] * src[i, j] + \
                       mask[5] * src[i, j + 1] + \
                       mask[6] * src[i + 1, j - 1] + \
                       mask[7] * src[i + 1, j] + \
                       mask[8] * src[i + 1, j + 1]

                if conv_type == ConvolutionType.MeanFilter:
                    conv = conv / 9
                    conv = 255 if conv > self.__threshold else 0
                elif conv_type == ConvolutionType.Sobel_X or conv_type == ConvolutionType.Sobel_Y:
                    conv = sqrt(pow(conv, 2))

                out[i, j] = self.__chk_val(conv)

    def enhancement(self, src, diff_2_img, mean_img):
        for i in range(1, diff_2_img.shape[0] - 1):
            for j in range(diff_2_img.shape[1] - 1):
                for c in range(3):
                    src[i, j, c] = self.__chk_val(src[i, j, c] + (diff_2_img[i, j] * (mean_img[i, j] / 255)))
        return src

    @staticmethod
    def __chk_val(v):
        if v > 255:
            return 255
        if v < 0:
            return 0
        return v


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("缺少要處理的圖片名稱")
        exit()
    main()
