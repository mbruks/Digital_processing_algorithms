import cv2
import math
import matplotlib.pyplot as plt
from scipy. special import logsumexp
import numpy as np

# Вывод картинок
def dis(imgarray):
    scale = 0.5

    rows = len(imgarray)  # Длина кортежа или списка
    cols = len(imgarray[
                   0])  # Если imgarray - это список, вернуть количество каналов первого изображения в списке, если это кортеж, вернуть длину первого списка, содержащегося в кортеже
    print("rows= ", rows, "cols=", cols)

    # Определить, является ли тип imgarray [0] списком
    # Список, указывающий, что imgarray является кортежем и должен отображаться вертикально
    rowsAvailable = isinstance(imgarray[0], list)

    # Ширина и высота первой картинки
    width = imgarray[0][0].shape[1]
    height = imgarray[0][0].shape[0]
    print("width=", width, "height=", height)

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

    cv2.namedWindow('Display window', cv2.WINDOW_NORMAL)
    cv2.imshow('Display window', ver)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Вывод двух картинок в сером оттенке
def plot_two_images_gray(img_1: np.array, img_2: np.array, title_1: str, title_2: str):

    plt.figure()

    ax1 = plt.subplot(1, 2, 1)
    plt.title(title_1)
    ax1.imshow(img_1, cmap='gray')

    ax2 = plt.subplot(1, 2, 2)
    plt.title(title_2)
    ax2.imshow(img_2, cmap='gray')

    plt.show()

# Вывод двух картинок
def plot_two_images(img_1: np.array, img_2: np.array, title_1: str, title_2: str):

    plt.figure()

    ax1 = plt.subplot(1, 2, 1)
    plt.title(title_1)
    ax1.imshow(img_1)

    ax2 = plt.subplot(1, 2, 2)
    plt.title(title_2)
    ax2.imshow(img_2)

    plt.show()

# Вычисление направлений/ориентаций
def calc_dir(filtered_x: np.array, filtered_y: np.array) -> np.array:
    height, width = filtered_x.shape

    theta = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            theta[i, j] = math.atan2(filtered_y[i, j], filtered_x[i, j])
            theta[i, j] = theta[i, j] * 180 / math.pi

    return theta

# Убираем отрицательное направление
def pos_dir(theta: np.array) -> np.array:
    height, width = theta.shape[0], theta.shape[1]

    for i in range(height):
        for j in range(width):
            theta[i, j] = 360 + theta[i, j] if theta[i, j] < 0 else theta[i, j]

    return theta

# Округляем до ближайшего угла в 0, 45, 90, или 135 градумов
def adjust_dir_nearest(theta: np.array) -> np.array:
    height, width = theta.shape

    theta_adj = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            if (theta[i, j] >= 0) and (theta[i, j] < 22.5) or (theta[i, j] >= 157.5) and (theta[i, j] < 202.5) or (
                    theta[i, j] >= 337.5) and (theta[i, j] <= 360):
                theta_adj[i, j] = 0
            elif (theta[i, j] >= 22.5) and (theta[i, j] < 67.5) or (theta[i, j] >= 202.5) and (theta[i, j] < 247.5):
                theta_adj[i, j] = 45
            elif (theta[i, j] >= 67.5) and (theta[i, j] < 112.5) or (theta[i, j] >= 247.5) and (theta[i, j] < 292.5):
                theta_adj[i, j] = 90
            elif (theta[i, j] >= 112.5) and (theta[i, j] < 157.5) or (theta[i, j] >= 292.5) and (theta[i, j] < 337.5):
                theta_adj[i, j] = 135

    return theta_adj


# Calculate Convolution's output size for one dimension
def calculate_target_size(img_size: int, kernel_size: int) -> int:
    num_pixels = 0

    for i in range(img_size):
        added = i + kernel_size
        num_pixels = num_pixels + 1 if added <= img_size else num_pixels

    return num_pixels

# Применение свертки к img
def convolve(img: np.array, kernel: np.array) -> np.array:
    # Получаем границы с учетом сверти
    height = calculate_target_size(
        img_size=img.shape[0],
        kernel_size=kernel.shape[0]
    )

    width = calculate_target_size(
        img_size=img.shape[1],
        kernel_size=kernel.shape[1]
    )

    k = kernel.shape[0]

    convolved_img = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            mat = img[i:i + k, j:j + k]
            convolved_img[i, j] = np.sum(np.multiply(mat, kernel))

    return convolved_img

# Матрица свертки
def gauss_filter(window_size: int, sigma: int) -> np.array:
    gauss_fil = np.zeros((window_size, window_size))
    for i in range(window_size):
        for j in range(window_size):
            gauss_fil[i, j] = np.exp((-i ** 2) / (2 * (sigma ** 2))) * np.exp((-j ** 2) / (2 * (sigma ** 2)))

    return gauss_fil

# Вычисление градиента
def calc_grad(filtered_x: np.array, filtered_y: np.array) -> np.array:
    grad = np.sqrt((filtered_x ** 2) + (filtered_y ** 2))

    return grad

# Подавление немаксимумов
def non_max_supr(grad: np.array, theta: np.array) -> np.array:

    height, width = grad.shape

    BW = np.zeros((height, width))

    for i in range(1, height - 1):
        for j in range(1, width - 1):
                # Влево или вправо
                if theta[i, j] == 0:
                    BW[i, j] = grad[i, j] == max([grad[i, j], grad[i, j+1], grad[i, j-1]])
                # Снизу слева или сверху справа
                elif theta[i, j] == 45:
                    BW[i, j] = grad[i, j] == max([grad[i, j], grad[i+1, j+1], grad[i-1, j-1]])
                # Вниз или вверх
                elif theta[i, j] == 90:
                    BW[i, j] = grad[i, j] == max([grad[i, j], grad[i+1, j], grad[i-1, j]])
                # Снизу слева или сверху справа
                elif theta[i, j] == 135:
                    BW[i, j] = grad[i, j] == max([grad[i, j], grad[i+1, j-1], grad[i-1, j+1]])

    return BW

# Пороговая фильтраци
def hysterisis_thresh(BW: np.array, t_low: int, t_high: int) -> np.array:


    height, width = BW.shape[0], BW.shape[1]

    t_res = np.zeros((height, width))

    for i in range(height - 1):
        for j in range(width - 1):
            t_res[i, j] = 1 if ((BW[i+1, j] > t_high and BW[i-1, j] > t_high) or
                                  (BW[i, j+1] > t_high and BW[i, j-1] > t_high) or
                                  (BW[i-1, j-1] > t_high and BW[i-1, j+1] > t_high) or
                                  (BW[i+1, j+1] > t_high and BW[i+1, j-1] > t_high)) or BW[i, j] > t_high else 0


    return t_res


def border_selection(img):
    # 1 шаг - Делаем черно-белые границы
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2 шаг Вычисляем Гаусовское размытие

    # Параметры фильтра Гаусса
    window_size = 5
    sigma = 5

    # Свертка изображения с помощью фильтра Гаусса
    img_conv = convolve(gray, gauss_filter(window_size, sigma))

    # Построение изображения в оттенках серого и отфильтрованного изображения
    plot_two_images_gray(gray, img_conv, "Grayscale Image", "Filtered Image")

    # 3 шаг Вычисление градиентов функции яркости

    # Определяем свертки
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Выбираем границы обхода
    # Свертка по изображению с помощью горизонтального и вертикального фильтров
    filtered_x = convolve(img_conv, Gx)
    filtered_y = convolve(img_conv, Gy)

    # Построение отфильтрованных изображений по осям X и Y
    plot_two_images_gray(filtered_x, filtered_y, "Filtered X-axis", "Filtered Y-axis")

    # Вычисляется градиент
    grad = calc_grad(filtered_x, filtered_y)

    # Получаем максимальный градиент
    max_grad = np.max(grad)

    # Вычисляем направление
    theta = calc_dir(filtered_x, filtered_y)

    # Корректировка для отрицательных направлений, делающая все направления положительными
    theta = pos_dir(theta)

    # Регулировка направления до ближайшего 0, 45, 90 или 135 градусов
    theta_adjusted = adjust_dir_nearest(theta)

    # plot_two_images(theta, theta_adjusted, "Directions", "Positive Directions")

    # Шаг 4 Подавление немаксимумов
    # (если значение градиента пикселя больше соседних, то пиксель определяется как граничный,
    # иначе значение пикселя подавляется)
    # ГРАНИЦЕЙ БУДЕТ СЧИТАТЬСЯ ПИКСЕЛЬ, ГРАДИЕНТ КОТОРОГО МАКСИМАЛЕН В СРАВНЕНИИ С ПИКСЕЛЯМИ ПО НАПРАВЛЕНИЮ НАИБОЛЬШЕГО РОСТА ФУНКЦИИ
    # ЕСЛИ ЗНАЧЕНИЕ ГРАДИЕНТА ВЫШЕ, ЧЕМ У ПИКСЕЛЕЙ СЛЕВА И СПРАВА, ТО ДАННЫЙ ПИКСЕЛЬ – ЭТО ГРАНИЦА, ИНАЧЕ – НЕ ГРАНИЦА.

    BW = non_max_supr(grad, theta_adjusted)

    # Убираем не границы
    dom_BW = np.multiply(grad, BW)

    # plot_two_images_gray(BW, dom_BW, "Edges\' grad", "Dominant Edges")

    # Шаг 5
    # Пороговая фильтрация(только границы) два порога: низ-выс значения градиента(процент от максимума) 20% 40%

    t_low = max_grad // 80
    t_high = max_grad // 10

    img_edges = 255 - (hysterisis_thresh(dom_BW, t_low, t_high) * 255)

    return img_edges


def main():
    img = cv2.imread(r'C:\Users\Admin\Desktop\spooky_house.jpg')

    # Внутренняя реализация
    # edges = cv2.Canny(img, 100, 400, L2gradient=False)

    edges = border_selection(img)

    cv2.imshow('Display window', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
