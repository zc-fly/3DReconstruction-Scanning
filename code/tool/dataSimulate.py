# -*- coding: utf-8 -*-
from libtiff import TIFF
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D


def simulation():
    """
    仿真了单个高斯荧光分子
    :return:
    """
    # function：  h(x,y) = bg + h*np.exp((-(X-x0)**2)/(2*wx**2))*np.exp((-(Y-y0)**2)/ (2*wy**2))
    bg = 100
    h = 3000
    x0 = 10.5
    y0 = 10.2
    wx = 4.0
    wy = 2.0
    x, y = range(20), range(20)
    X, Y = np.meshgrid(x, y)
    # X-=0.5
    # Y-=0.5

    # 是否旋转矩阵
    isRatate = False

    Z = bg + h * np.exp((-(X - x0) ** 2) / (2 * wx ** 2)) * np.exp((-(Y - y0) ** 2) / (2 * wy ** 2))

    ## 矩阵旋转
    if isRatate:
        rows, cols = Z.shape[:2]
        # 第一个参数旋转中心，第二个参数旋转角度，第三个参数：缩放比例
        M1 = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
        # 第三个参数：变换后的图像大小
        Z = cv2.warpAffine(Z, M1, (rows, cols))

    # 保存
    cv2.imwrite('img.tif', Z.astype(np.float32))

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow', alpha=0.9)
    plt.show()


def data_simulate():
    """
    仿真了两个分子点不同距离的合成图像
    :return:
    """
    path = r'D:\\A_CodeFile\\python\\meifishReconstruct3D\\data\\xyz_target'
    Map_set = 200 #80*80pixel正方形
    distanceContral_x = 0 #pixel
    distanceContral_y = 12  # pixel
    noiseDev = 5

    tif1 = TIFF.open(path + '\\' + 'point1.tif', mode='r')
    tif2 = TIFF.open(path + '\\' + 'point2.tif', mode='r')
    img_stack1 = list()
    img_stack2 = list()
    img_stack = list()
    for img in list(tif1.iter_images()):
        img_stack1.append(img)
    for img in list(tif2.iter_images()):
        img_stack2.append(img)
    n1, m1 = img_stack1[0].shape
    n2, m2 = img_stack2[0].shape

    for i in range(0, len(img_stack1)):
        background = min(img_stack1[i].min(), img_stack2[i].min())
        Map = np.random.normal(background, noiseDev, size=(Map_set, Map_set))
        Map[Map_set//2 : Map_set//2 + n1, Map_set//2 : Map_set//2 + m1] = Map[Map_set//2 : Map_set//2 + n1, Map_set//2 : Map_set//2 + m1] + img_stack1[i] - background
        Map[Map_set//2+distanceContral_x : Map_set//2+n2+distanceContral_x, Map_set//2+distanceContral_y : Map_set//2+m2+distanceContral_y] = \
            Map[Map_set//2+distanceContral_x : Map_set//2+n2+distanceContral_x, Map_set//2+distanceContral_y : Map_set//2+m2+distanceContral_y] + img_stack2[i] - background
        img_stack.append(Map)

    savetif = TIFF.open(path + '\\' + 'point_simulate.tif', mode='w')
    for img in img_stack:
        img = Image.fromarray(img)
        savetif.write_image(img)
    savetif.close()