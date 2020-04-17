import imageio
import numpy as np
from PIL import Image


def get_image_data(path):
    im = Image.open(path)
    rgb = np.array(im)
    return im, rgb

def show_image(data):
    pilImg = Image.fromarray(np.uint8(data))
    pilImg.show()

def compression_data_x(data, scale):
    ans = []
    for row in rgb:
        tmp_row = np.array([[0, 0, 0]])
        count = 0
        sum_col = np.zeros(3)
        for col in row:
            sum_col += col
            if count == scale:
                tmp_row = np.append(tmp_row, np.array([sum_col/scale]), axis=0)
                count = 0
                sum_col = np.zeros(3)
            count += 1
        ans.append(tmp_row[1:])
    # show_image(np.array(ans))
    return np.array(ans)

def compression_data_y(data, scale):
    ans = []
    height, width = data.shape[0], data.shape[1]
    print(height, width)
    count = 0
    tmp = np.array([np.array([0, 0, 0], dtype=np.float64) for i in range(width)], dtype=np.float64)
    for index in range(height):
        if count == scale:
            ans.append(tmp / scale)
            # 初期化
            tmp = np.array([np.array([0, 0, 0], dtype=np.float64) for i in range(width)], dtype=np.float64)
            count = 0
        else:
            tmp = tmp + data[index]
        count += 1
    show_image(np.array(ans))
    return np.array(ans)

# 縮小
def get_small_data(rgb, scale, average=False):
    if average:
        if scale == 1:
            return rgb
        else:
            # 横に圧縮
            ans = compression_data_x(rgb, scale)
            # 縦に圧縮
            ans = compression_data_y(ans, scale)
            return ans
    else:
      return rgb[::scale, ::scale]

# 拡大
def get_big_data(rgb, scale, average=False):
    pass

# 量子化
def make_quantization(signal, bit_length):
    quantized_signal = signal * ((2 ** (bit_length / 2)) - 1) # プラスで8bit，マイナスで8bit
    quantized_signal = np.floor(quantized_signal) # 小数部切り捨て
    quantized_signal = quantized_signal / np.max(np.abs(quantized_signal)) # [-1, 1]の区間に
    return quantized_signal

if __name__ == '__main__':
    path = './image.jpg'
    im, rgb = get_image_data(path)
    width, height = im.size
    scale = 1
    small_data = get_small_data(rgb, scale=scale, average=True)
    pilImg = Image.fromarray(np.uint8(small_data))
    pilImg.save('image_small_{}.jpg'.format(scale))
    # pilImg.show()

    # 縮小
    # width_small, height_small = width // 2, height // 2
    # im_small = im.resize(size=(width_small, height_small))
    # im_small.save('image_small.jpg')
