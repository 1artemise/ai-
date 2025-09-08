#!/usr/bin/env python
# coding=utf-8
# @Software: PyCharm

from PIL import Image


def make_ico_file(src_image_file, dist_ico_file, size_list=None):
    """
    :param src_image_file:
    :param dist_ico_file:
    :return:
    """
    default_size_list = [
        (256, 256),
        (128, 128),
        (64, 64),
        (48, 48),
        (32, 32),
        (24, 24),
        (16, 16)
    ]
    size_list = size_list or default_size_list
    image = Image.open(src_image_file)
    image_cropped = image.crop((0, 0, 256, 256))
    image_cropped.save(dist_ico_file, sizes=size_list)


if __name__ == '__main__':
    make_ico_file(src_image_file="D:\\fool\\Medical-Report-Generation-master\\pytorch_flask_service\\cat.png",
                  dist_ico_file='cat.ico')