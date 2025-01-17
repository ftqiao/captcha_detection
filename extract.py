# -*- coding: utf-8 -*-
# @Time    : 2020/11/12 22:54
# @Author  : AaronJny
# @File    : extract.py
# @Desc    :
import os
from glob import glob

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def extract_correct_word():
    save_dir = 'D:\\yntrust\\PycharmProjects\\captcha_detection\\correct_words'
    # save_dir = '/mnt/captcha_detection/correct_words'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    images = glob('D:\\yntrust\\PycharmProjects\\captcha_detection\\images\\*.png')
    # images = glob('/mnt/captcha_detection/images/*.png')
    for image_path in tqdm(images):
        # 获取图片编号
        num = image_path.split('\\')[-1].split('.')[0]
        # num = image_path.split('/')[-1].split('.')[0]
        # 读取图片
        im = np.asarray(Image.open(image_path).convert("RGB"))
        # 切割第一个字
        word_im = im[1:34, 186:216]
        cv2.imwrite(os.path.join(save_dir, 'x-{}-{}.png'.format(num, 0)), word_im)
        # 切割第二个字
        word_im = im[1:34, 216:246]
        cv2.imwrite(os.path.join(save_dir, 'x-{}-{}.png'.format(num, 1)), word_im)


def extract_gen_word():
    # 创建目录
    save_dir = 'D:\\yntrust\\PycharmProjects\\captcha_detection\\gen_words'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # 记录读取过的编号，保存到文件，中断操作时，跳过处理过的图片
    dealImg_path = 'D:\\yntrust\\PycharmProjects\\captcha_detection\\dealImg.txt'
    with open('dealImg.txt', 'r') as f:
        imgs = f.read().splitlines()
    # 读取数据标注结果
    with open('xyolo_label.txt', 'r') as f:
        lines = f.readlines()
    # 对于每一张图片
    for line in lines:
        n_line = line.strip()
        image_path, *pos = n_line.split()
        num = image_path.split('\\')[-1].split('.')[0]
        if num in imgs:
            continue
        im = np.asarray(Image.open(image_path).convert("RGB"))
        # 对于每一个框框
        for index, _pos in enumerate(pos):
            x1, y1, x2, y2, cls = map(int, _pos.split(','))
            n_im = im.copy()
            cv2.rectangle(n_im, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow('text', n_im)
            cv2.waitKey(1000)
            word = input('图片{}请输入当前选中汉字：'.format(num))
            # cv2.imwrite(os.path.join('D:/yntrust/PycharmProjects/captcha_detection/gen_words/{}-{}-{}.png'.format(word, num, index)), im[y1:y2, x1:x2])
            filename = 'D:/yntrust/PycharmProjects/captcha_detection/gen_words/{}-{}-{}.png'.format(word, num, index)
            cv2.imencode(filename, im[y1:y2, x1:x2])[1].tofile(filename)
            del n_im
            cv2.destroyAllWindows()
        with open(dealImg_path, 'a', encoding='utf8') as f:
            f.write(num + '\n')


if __name__ == '__main__':
    # extract_correct_word()
    extract_gen_word()
