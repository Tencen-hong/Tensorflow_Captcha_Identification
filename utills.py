# coding=utf-8
from param import *


def gen_captcha_text_and_image():
    """
    :return:验证码的label和图片
    """

    def random_captcha_text():
        """
        从字符集中找（指定个数[param.py]的）元素生成验证码
        :return:待生成的验证码元素
        """

        captcha_text = []
        for i in range(MAX_CAPTCHA_LEN):
            c = random.choice(char_set)
            captcha_text.append(c)
        return captcha_text

    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)  # 转换成字符串
    captcha = ImageCaptcha().generate(captcha_text)
    captcha_image = Image.open(captcha)
    return captcha_text, captcha_image


def txt2vec(text):
    """
    :return:验证码字符转成的向量
    """
    vector = np.zeros([MAX_CAPTCHA_LEN, CHAR_SET_LEN])
    for i, c in enumerate(text):
        if c.isdigit():
            vector[i, int(c)] = 1  # 数字对应位标记1
        elif c.isalpha() and c.islower():
            vector[i, ord(c) - 97 + 10] = 1  # 小写字母对应位标记1
        elif c.isalpha() and c.isupper():
            vector[i, ord(c) - 65 + 26 + 10] = 1  # 大写字母对应位标记1
    return vector


def vec2text(vec):
    char_pos = vec
    text = []
    for i, c in enumerate(char_pos):
        char_at_pos = i  # c/63
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('a')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('A')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)


def gen_batch(batch_size):
    """

    :param batch_size:
    :return: batch_x, batch_y
    """
    batch_x = np.zeros([batch_size, 60 * 160], np.float32)
    batch_y = np.zeros([batch_size, MAX_CAPTCHA_LEN * CHAR_SET_LEN], np.int32)
    for i in range(batch_size):
        text, image = gen_captcha_text_and_image()
        image = image.convert('L')
        gray = np.array(image)

        vector = txt2vec(text)
        batch_x[i, :] = gray.flatten() / 255  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = vector.reshape([-1, MAX_CAPTCHA_LEN * CHAR_SET_LEN])

    batch_x = batch_x.reshape(-1, 60, 160, 1)
    return batch_x, batch_y


if __name__ == '__main__':
    text, image = gen_captcha_text_and_image()
    image = image.convert('L')  # 转灰度

    gray = np.array(image)  # 转换成numpy矩阵
    f = plt.figure()
    ax = f.add_subplot(111)  # 创建并选择子图
    ax.text(0.1, 0.9, text, transform=ax.transAxes)
    plt.imshow(gray)
    plt.show()
