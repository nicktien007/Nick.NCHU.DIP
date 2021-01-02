import cv2
import numpy as np
import time


def resize(src, new_size):
    dst_w, dst_h = new_size  # 目標圖像寬高
    src_h, src_w = src.shape[:2]  # 源圖像寬高
    if src_h == dst_h and src_w == dst_w:
        return src.copy()
    scale_x = float(src_w) / dst_w  # x縮放比例
    scale_y = float(src_h) / dst_h  # y縮放比例

    # 遍歷目標圖像，插值
    dst = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    for n in range(3):  # 對channel循環
        for dst_y in range(dst_h):  # 對height循環
            for dst_x in range(dst_w):  # 對width循環
                # 目標在源上的坐標
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5
                # 計算在源圖上四個近鄰點的位置
                src_x_0 = int(np.floor(src_x))
                src_y_0 = int(np.floor(src_y))
                src_x_1 = min(src_x_0 + 1, src_w - 1)
                src_y_1 = min(src_y_0 + 1, src_h - 1)

                # 雙線性插值
                value0 = (src_x_1 - src_x) * src[src_y_0, src_x_0, n] + (src_x - src_x_0) * src[src_y_0, src_x_1, n]
                value1 = (src_x_1 - src_x) * src[src_y_1, src_x_0, n] + (src_x - src_x_0) * src[src_y_1, src_x_1, n]
                dst[dst_y, dst_x, n] = int((src_y_1 - src_y) * value0 + (src_y - src_y_0) * value1)
    return dst


if __name__ == '__main__':
    img_in = cv2.imread('./test/Lena_2.png')
    start = time.time()
    # img_out = cv2.resize(img_in, (800, 800))
    img_out = resize(img_in, (1000, 1000))
    print('cost %f seconds' % (time.time() - start))

    cv2.imshow('src_image', img_in)
    cv2.imshow('dst_image', img_out)
    cv2.waitKey()
