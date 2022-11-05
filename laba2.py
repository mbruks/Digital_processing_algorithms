# import math
# import cv2
# import numpy as np
#
#
# img1 = cv2.imread(r'C:\Users\Admin\Desktop\spooky_house.jpg')
# newGray = cv2.imread(r'C:\Users\Admin\Desktop\spooky_house.jpg')
# gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Display window', gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# h, v = gray.shape[:2]
# sum = 0
# n = 5
# p = n//2
# gauss_matrix = [[0] * n for i in range(n)]
# for k in range(n):
#     for l in range(n):
#         gauss_matrix[k][l] = (1 / (2 * math.pi)) * np.exp(-((k - p) ** 2 + (l - p) ** 2) / 2)
# for k in range(n):
#     for l in range(0, n):
#         sum += gauss_matrix[k][l]
# print(sum)
# print(gauss_matrix)
#
# new_sum = 0
# for k in range(n):
#     for l in range(n):
#         gauss_matrix[k][l] = gauss_matrix[k][l] / sum
#         new_sum += gauss_matrix[k][l]
# print(new_sum)
#
# print(gauss_matrix)
# finishh = h - p
# finishv = v - p
#
# for i in range(p, finishh):
#     for j in range(p, p):
#         new_value = 0
#         for k in range(n):
#             for l in range(n):
#                 new_value = new_value + gauss_matrix[k][l] * gray[i - p + k][j - p + l]
#         newGray[i][j] = new_value
# cv2.namedWindow('old', cv2.WINDOW_AUTOSIZE)
# cv2.imshow('old', gray)
# cv2.namedWindow('new', cv2.WINDOW_AUTOSIZE)
# cv2.imshow('new', newGray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np

# Определите функцию, первый параметр - это коэффициент масштабирования,
# а второй параметр - это кортеж или список изображений, которые будут отображаться
def ManyImgs(scale, imgarray):
    rows = len(imgarray)  # Длина кортежа или списка
    cols = len(imgarray[0])
    # Если imgarray - это список, вернуть количество каналов первого изображения в списке, если это кортеж, вернуть длину первого списка, содержащегося в кортеже
    # print("rows=", rows, "cols=", cols)

    # Определить, является ли тип imgarray [0] списком
    # Список, указывающий, что imgarray является кортежем и должен отображаться вертикально
    rowsAvailable = isinstance(imgarray[0], list)

    # Ширина и высота первой картинки
    width = imgarray[0][0].shape[1]
    height = imgarray[0][0].shape[0]
    # print("width=", width, "height=", height)

    # Если входящий кортеж
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                # Обойти кортеж, если это первое изображение, не преобразовывать
                if imgarray[x][y].shape[:2] == imgarray[0][0].shape[:2]:
                    imgarray[x][y] = cv2.resize(imgarray[x][y], (0, 0), None, scale, scale)
                # Преобразуйте другие матрицы к тому же размеру, что и первое изображение, и коэффициент масштабирования будет масштабироваться
                else:
                    imgarray[x][y] = cv2.resize(imgarray[x][y], (imgarray[0][0].shape[1], imgarray[0][0].shape[0]),
                                                None, scale, scale)
                # Если изображение в оттенках серого, преобразовать его в цветное отображение
                if len(imgarray[x][y].shape) == 2:
                    imgarray[x][y] = cv2.cvtColor(imgarray[x][y], cv2.COLOR_GRAY2BGR)

        # Создайте пустой холст того же размера, что и первое изображение
        imgBlank = np.zeros((height, width, 3), np.uint8)
        hor = [
                  imgBlank] * rows  # Тот же размер, что и первое изображение, и столько же горизонтальных пустых изображений, сколько кортеж содержит список
        for x in range(0, rows):
            # Расположить x-й список в кортеже по горизонтали
            hor[x] = np.hstack(imgarray[x])
        ver = np.vstack(hor)  # Объединить разные списки по вертикали
    # Если входящий - это список
    else:
        # Операция трансформации, как и раньше
        for x in range(0, rows):
            if imgarray[x].shape[:2] == imgarray[0].shape[:2]:
                imgarray[x] = cv2.resize(imgarray[x], (0, 0), None, scale, scale)
            else:
                imgarray[x] = cv2.resize(imgarray[x], (imgarray[0].shape[1], imgarray[0].shape[0]), None, scale, scale)
            if len(imgarray[x].shape) == 2:
                imgarray[x] = cv2.cvtColor(imgarray[x], cv2.COLOR_GRAY2BGR)
        # Расположить список по горизонтали
        hor = np.hstack(imgarray)
        ver = hor
    return ver


def GF(img, size, sigma):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    newGray = img
    newGray = cv2.cvtColor(newGray, cv2.COLOR_BGR2GRAY)
    h, w = newGray.shape[:2]
    n = size
    pad = n // 2
    g_m = [[0] * n for i in range(n)]

    summ = 0
    for i in range(n):
        for j in range(n):
            g_m[i][j] = (1 / (2 * np.pi * sigma ** 2) * np.exp(
                -((i - pad) ** 2 + (j - pad) ** 2) / (2 * sigma ** 2)))
            summ += g_m[i][j]

    print(summ)
    summ_2 = 0
    for i in range(n):
        for j in range(n):
            g_m[i][j] = g_m[i][j] / summ
            summ_2 += g_m[i][j]

    finishh = h - pad
    finishw = w - pad

    for i in range(pad, finishh):
        for j in range(pad, finishw):
            new_value = 0
            for k in range(n):
                for l in range(n):
                    new_value = new_value + g_m[k][l] * gray[i - pad + k][j - pad + l]
            newGray[i][j] = new_value

    return newGray


#   Реализовать фильтр Гаусса средствами языка python.
def task_1(size, sigma):
    image = cv2.imread(r'C:\Users\Admin\Desktop\spooky_house.jpg')

    stackedimageb = ManyImgs(0.7, ([image, GF(image, size, sigma)]))

    cv2.namedWindow("Window")
    cv2.imshow("Window", stackedimageb)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


#   Применить данный фильтр для двух разных значений
#   среднего квадратичного отклонения и двух разных размерностей матрицы
#   свертки, сравнить результаты для ОДНОГО изображения.
def task_2(size, sigma):
    image = cv2.imread(r'C:\Users\Admin\Desktop\spooky_house.jpg')
    image_blur_3 = GF(image, size, sigma)
    image_blur_5 = GF(image, size + 2, sigma + 0.1)
    image_blur_11 = GF(image, size + 4, sigma + 0.2)

    stackedimageb = ManyImgs(0.7, ([image, image_blur_3], [image_blur_5, image_blur_11]))

    cv2.namedWindow("Window")
    cv2.imshow("Window", stackedimageb)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


#   Реализовать размытие Гаусса встроенным методом
#   библиотеки OpenCV, сравнить результаты с Вашей реализацией.
def task_3(size, sigma):
    image = cv2.imread(r'C:\Users\Admin\Desktop\spooky_house.jpg')
    img_blur_7 = cv2.GaussianBlur(image, (size, size), sigma)
    image_blur_7 = GF(image, size, sigma)

    Blankimg = np.zeros((200, 200), np.uint8)
    stackedimageb = ManyImgs(0.7, ([image, Blankimg], [img_blur_7, image_blur_7]))

    cv2.namedWindow("Window")
    cv2.imshow("Window", stackedimageb)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    size = 5
    sigma = 1.8
    #task_1(size, sigma)
    #task_2(size, sigma)
    #task_3(size, sigma)