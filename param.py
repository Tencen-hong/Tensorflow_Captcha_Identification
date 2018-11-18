# coding=utf-8
from captcha.image import ImageCaptcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import os
import time

path = "img"
log_path = "logs/"

def mkdir(path):
    """
    创建image目录
    :param path:
    :return:
    """
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---", path, "---  OK  ---")
    else:
        print("---  There is this folder!  ---")


def the_current_time():
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time()))),end=" : ")


number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
            'u', 'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z']

# char_set = number
char_set = number + alphabet + ALPHABET
MAX_CAPTCHA_LEN = 4  # 最大验证码长度
CHAR_SET_LEN = len(char_set)  # 字符集数量

