import numpy as np

default_structure = np.ones((3, 3))


def idx_check(index):
    if index < 0:
        return 0
    else:
        return index


def dilation(binary_img_matrix=None, structuring_element=default_structure):
    binary_img_matrix = np.asarray(binary_img_matrix)
    structuring_element = np.asarray(structuring_element)
    ste_shp = structuring_element.shape
    dilated_img = np.zeros((binary_img_matrix.shape[0], binary_img_matrix.shape[1]))
    ste_origin = (int((structuring_element.shape[0] - 1) / 2),int((structuring_element.shape[1] - 1) / 2))

    for i in range(len(binary_img_matrix)):
        for j in range(len(binary_img_matrix[0])):
            overlap = binary_img_matrix[idx_check(i - ste_origin[0]):i + (ste_shp[0] - ste_origin[0]),
                      idx_check(j - ste_origin[1]):j + (ste_shp[1] - ste_origin[1])]
            shp = overlap.shape

            ste_first_row_idx = int(np.fabs(i - ste_origin[0])) if i - ste_origin[0] < 0 else 0
            ste_first_col_idx = int(np.fabs(j - ste_origin[1])) if j - ste_origin[1] < 0 else 0

            ste_last_row_idx = ste_shp[0] - 1 - (i + (ste_shp[0] - ste_origin[0]) - binary_img_matrix.shape[0]) if i + (
                        ste_shp[0] - ste_origin[0]) > binary_img_matrix.shape[0] else ste_shp[0] - 1
            ste_last_col_idx = ste_shp[1] - 1 - (j + (ste_shp[1] - ste_origin[1]) - binary_img_matrix.shape[1]) if j + (
                        ste_shp[1] - ste_origin[1]) > binary_img_matrix.shape[1] else ste_shp[1] - 1

            if shp[0] != 0 and shp[1] != 0 and np.logical_and(
                    structuring_element[ste_first_row_idx:ste_last_row_idx + 1, ste_first_col_idx:ste_last_col_idx + 1],
                    overlap).any():
                dilated_img[i, j] = 1
    return dilated_img


def erosion(binary_img_matrix=None, structuring_element=default_structure):
    binary_img_matrix = np.asarray(binary_img_matrix)
    structuring_element = np.asarray(structuring_element)
    ste_shp = structuring_element.shape
    eroded_img = np.zeros((binary_img_matrix.shape[0], binary_img_matrix.shape[1]))
    ste_origin = (
    int(np.ceil((structuring_element.shape[0] - 1) / 2.0)), int(np.ceil((structuring_element.shape[1] - 1) / 2.0)))
    for i in range(len(binary_img_matrix)):
        for j in range(len(binary_img_matrix[0])):
            overlap = binary_img_matrix[idx_check(i - ste_origin[0]):i + (ste_shp[0] - ste_origin[0]),
                      idx_check(j - ste_origin[1]):j + (ste_shp[1] - ste_origin[1])]
            shp = overlap.shape
            ste_first_row_idx = int(np.fabs(i - ste_origin[0])) if i - ste_origin[0] < 0 else 0
            ste_first_col_idx = int(np.fabs(j - ste_origin[1])) if j - ste_origin[1] < 0 else 0

            ste_last_row_idx = ste_shp[0] - 1 - (i + (ste_shp[0] - ste_origin[0]) - binary_img_matrix.shape[0]) if i + (
                        ste_shp[0] - ste_origin[0]) > binary_img_matrix.shape[0] else ste_shp[0] - 1
            ste_last_col_idx = ste_shp[1] - 1 - (j + (ste_shp[1] - ste_origin[1]) - binary_img_matrix.shape[1]) if j + (
                        ste_shp[1] - ste_origin[1]) > binary_img_matrix.shape[1] else ste_shp[1] - 1

            if shp[0] != 0 and shp[1] != 0 and np.array_equal(
                    np.logical_and(overlap, structuring_element[ste_first_row_idx:ste_last_row_idx + 1,
                                            ste_first_col_idx:ste_last_col_idx + 1]),
                    structuring_element[ste_first_row_idx:ste_last_row_idx + 1,
                    ste_first_col_idx:ste_last_col_idx + 1]):
                eroded_img[i, j] = 1
    return eroded_img
