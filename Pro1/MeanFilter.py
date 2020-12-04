#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/3 21:34
# @Author  : Dodo
# @File    : MeanFilter.py
# @Software: PyCharm
# @Contact : gr_yao@126.com
# @Function: Perform the average filter using the integral graph algorithm.
import cv2
import numpy as np


def generate_integral_image(img):
    """
    This function generates an integral graph of the image.

    :param img: The image of the integral diagram needs to be generated.
    :return: Integral diagram of the image.
    """
    # Get image size.
    h = img.shape[0]
    w = img.shape[1]

    # Initialize the generated image.
    b = np.zeros((h, w), dtype=int)

    # Integral operation.
    for i in range(h):
        for j in range(w):
            b[i, j] = np.sum(img[0: i + 1, 0: j + 1])
    return b


def mean_filter_2d(int_img, ker_size=3):
    """
    This function can make use of the integral image of the image for fast mean filtering.

    :param int_img: The integral image of the image to be filtered by means.
    :param ker_size: The window size of mean filtering.
    :return: The image obtained by mean filtering.
    """
    # Fill the edge of the image to ensure that
    # the output image is the same size as the original.
    pad_h = ker_size // 2
    pad_w = ker_size // 2
    int_img = np.pad(int_img, (pad_h, pad_w), 'edge')

    # The width and height of the output image.
    h = int_img.shape[0] - ker_size + 1
    w = int_img.shape[1] - ker_size + 1

    # Initialize the result image.
    res = np.zeros((h, w))

    # Filtering operation.
    for i in range(h):
        for j in range(w):
            res[i, j] = (int_img[i, j] +
                         int_img[i + ker_size - 1, j + ker_size - 1] -
                         int_img[i, j + ker_size - 1] -
                         int_img[i + ker_size - 1, j]) / (ker_size ** 2)

    # Return the result image.
    return res


if __name__ == "__main__":
    image = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (image.shape[1] // 10, image.shape[0] // 10))
    cv2.imwrite("image.jpg", image)
    integral_image = generate_integral_image(image)
    result = mean_filter_2d(integral_image, 5)
    cv2.imwrite("result2.jpg", result)
