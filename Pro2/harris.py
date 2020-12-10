#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/4 17:49
# @Author  : Dodo
# @File    : harris.py
# @Software: PyCharm
# @Contact : gr_yao@126.com
# @Function: 
import cv2
import numpy as np
from matplotlib import pylab as plt


def img_rot(img, a=30):
    (h, w) = img.shape[:2]
    r = cv2.getRotationMatrix2D(center=(w / 2, h / 2), angle=a, scale=1.0)
    res = cv2.warpAffine(img, r, (w, h))
    return res


def harris_det(img, k=0.04, th=0.1):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    Ix = cv2.Scharr(g, -1, 1, 0)
    Iy = cv2.Scharr(g, -1, 0, 1)

    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy

    Ix2 = cv2.GaussianBlur(Ix2, (5, 5), 0)
    Iy2 = cv2.GaussianBlur(Iy2, (5, 5), 0)
    Ixy = cv2.GaussianBlur(Ixy, (5, 5), 0)

    # prepare output image
    g = np.array((g, g, g))
    g = np.transpose(g, (1, 2, 0))

    # get R
    r = (Ix2 * Iy2 - Ixy ** 2) - k * ((Ix2 + Iy2) ** 2)

    # detect corner
    img[r >= np.max(r) * th] = [255, 0, 0]

    res = img.astype(np.uint8)

    return res


if __name__ == "__main__":
    image = cv2.imread("test.png")
    # rot_img = img_rot(image)
    rot_img = harris_det(image)
    plt.imshow(rot_img)
    plt.show()
