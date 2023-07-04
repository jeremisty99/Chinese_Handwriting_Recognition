from PIL import Image
from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt


# 没什么用的MSER
def locate_process_mser(img):
    """
        :param img: 图
    """
    origin_img = img.copy()
    # img = cv2.resize(img, (64, 64))
    # 先开运算(腐蚀再膨胀)，让文字部首间联通
    kernel = np.ones((10, 10), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mser = cv2.MSER_create()
    regions, boxes = mser.detectRegions(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    offset = 0
    position_list = []
    for box in boxes:
        x, y, w, h = box
        print(x, y, w, h)
        # if w / h > 1.5 or h / w > 1.5:
        #     continue
        position_list.append((x, y, w, h))
    position_list = list(set(position_list))
    print(position_list)
    x_array = [i[0] for i in position_list]
    x_array.remove(max(x_array))
    # x_array.remove(min(x_array))
    x = int(np.mean(x_array))

    y_array = [i[1] for i in position_list]
    y_array.remove(max(y_array))
    y_array.remove(min(y_array))
    y = int(np.mean(y_array))

    w_array = [i[2] for i in position_list]
    w_array.remove(max(w_array))
    w_array.remove(min(w_array))
    w = int(np.mean(w_array))

    h_array = [i[3] for i in position_list]
    h_array.remove(max(h_array))
    h_array.remove(min(h_array))
    h = int(np.mean(h_array))
    cv2.rectangle(img, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 0), 1)

    new_img = origin_img[(y - offset):(y + h + offset), (x - offset): (x + w + offset)]
    plt.imshow(img, "gray")
    # plt.imshow(new_img, "gray")
    plt.show()
    return Image.fromarray(new_img)


# 水平方向投影
def h_project(binary):
    h, w = binary.shape

    # 水平投影
    hprojection = np.zeros(binary.shape, dtype=np.uint8)

    # 创建h长度都为0的数组
    h_h = [0] * h
    for j in range(h):
        for i in range(w):
            if binary[j, i] == 0:
                h_h[j] += 1
    # 画出投影图
    for j in range(h):
        for i in range(h_h[j]):
            hprojection[j, i] = 255

    # plt.imshow(hprojection)
    # plt.show()
    return h_h


# 垂直反向投影
def v_project(binary):
    h, w = binary.shape
    # 垂直投影
    vprojection = np.zeros(binary.shape, dtype=np.uint8)

    # 创建 w 长度都为0的数组
    v_v = [0] * w
    for i in range(w):
        for j in range(h):
            if binary[j, i] == 0:
                v_v[i] += 1

    for i in range(w):
        for j in range(v_v[i]):
            vprojection[j, i] = 255

    # plt.imshow(vprojection)
    # plt.show()
    return v_v


def locate_process_projection_single(img):
    origin_img = img.copy()
    # 可选
    # img = cv2.resize(img, (500, 200), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    # plt.imshow(gray)
    # plt.show()
    # np.set_printoptions(threshold=np.inf)
    # print(gray)
    h, w = gray.shape
    h_h = h_project(gray)
    v_v = v_project(gray)
    y_begin = (next((i for i, x in enumerate(h_h) if x), None))
    h_h.reverse()
    y_end = h - (next((i for i, x in enumerate(h_h) if x), None))

    x_begin = (next((i for i, x in enumerate(v_v) if x), None))
    v_v.reverse()
    x_end = w - (next((i for i, x in enumerate(v_v) if x), None))

    # print(x_begin, y_begin, x_end, y_end)

    # cv2.rectangle(img, (x_begin, y_begin), (x_end, y_end), (0, 0, 255), 1)
    new_img = origin_img[y_begin:y_end, x_begin:x_end]
    # plt.imshow(new_img, "gray")
    # plt.show()
    return Image.fromarray(new_img)


def locate_process_projection_multiple(img):
    h, w, _ = img.shape
    w = int(w * 128 / h)
    h = 128
    img = cv2.resize(img, (w, 128), interpolation=cv2.INTER_CUBIC)
    # print(img.shape)
    # plt.imshow(img)
    # plt.show()
    origin_img = img.copy()
    rectangle_img = img.copy()
    kernel = np.ones((10, 10), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 水平投影分行
    h_h = h_project(gray)
    # print(len(h_h))
    start = 0
    h_start, h_end = [], []
    position = []
    # print(h_h)
    start_flag = 0
    for i in range(len(h_h)):
        if h_h[i] > 0 and start_flag == 0:
            h_start.append(i)
            start_flag = 1
        if sum(h_h[i - 3:i + 3]) == 0 and start_flag == 1:
            h_end.append(i)
            start_flag = 0
    if start_flag == 1:
        h_end.append(len(h_h))
    # print(h_start)
    # print(h_end)

    for i in range(len(h_start)):
        cropimg = gray[h_start[i]:h_end[i], 0:w]
        v_v = v_project(cropimg)
        w_start_flag = 0
        w_start = 0
        for j in range(len(v_v)):
            if v_v[j] > 0 and w_start_flag == 0:
                w_start = j
                w_start_flag = 1
            if sum(v_v[j - 5:j + 5]) == 0 and w_start_flag == 1:
                w_end = j
                w_start_flag = 0
                position.append((w_start, h_start[i], w_end, h_end[i], i))
    # print(position)
    # 确定分割位置
    # print(p)
    # area = abs(p[1] - p[0]) * abs(p[3] - p[2])
    # print(area)
    img_list = []
    a = 0
    for p in position:
        cv2.rectangle(rectangle_img, (p[0], p[1]), (p[2], p[3]), (0, 0, 255), 1)
        new_img = origin_img[p[1]:p[3], p[0]:p[2]]
        # cv2.imwrite(str(a) + ".jpg", new_img)
        # a = a + 1
        # plt.imshow(new_img, "gray")
        # plt.show()
        img_list.append((Image.fromarray(new_img), p[4]))
        # plt.imshow(new_img)
        # plt.show()
    # plt.imshow(rectangle_img)
    # plt.show()
    return img_list, rectangle_img


if __name__ == '__main__':
    locate_process_projection_multiple(cv2.imread('1.jpg'))
