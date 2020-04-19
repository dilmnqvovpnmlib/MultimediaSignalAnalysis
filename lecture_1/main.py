import imageio
import numpy as np
from PIL import Image


class ImageProcessing:
    def __init__(self, path):
        self.im, self.rgb = self.get_image_data(path)
        self.show(self.rgb, 'normal', 255)

    def get_image_data(self, path):
        im = Image.open(path)
        rgb = np.array(im)
        return im, rgb
    
    def show(self, data, type, scale):
        pilImg = Image.fromarray(np.uint8(data))
        pilImg.show()
        pilImg.save('./{0}/{0}_{1}.jpg'.format(type, scale))
        return pilImg
    
    # 拡大
    def get_big_data(self, scale):
        data = self.rgb.repeat(scale, axis=0).repeat(scale, axis=1)
        self.show(data, 'big', scale)
        return data
    
    """
    x軸方向に縮小
    """
    def compression_data_x(self, scale):
        ans = []
        for row in self.rgb:
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
        return np.array(ans)

    """
    y軸方向に縮小
    """
    def compression_data_y(self, data, scale):
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
        return np.array(ans)

    # 縮小
    def get_small_data(self, scale, average=False):
        if average:
            if scale == 1:
                self.show(self.rgb, 'small', scale)
                return rgb
            else:
                # 横に圧縮
                ans = self.compression_data_x(scale)
                # 縦に圧縮
                data = self.compression_data_y(ans, scale)
                self.show(data, 'small', scale)
                return data
        else:
            # 間引く方法
            data = self.rgb[::scale, ::scale]
            self.show(data, 'small', -1)
            return data

    # 量子化
    def get_quantized_data(self, bit_length):
        q = 255 / (2 ** bit_length)
        data = q * np.round(self.rgb / q)
        self.show(data, 'quantized', bit_length)
        return data

if __name__ == '__main__':
      # path = './image.jpg'
      path = './IE.jpeg'
      image_processing = ImageProcessing(path)
      
      # big_scale = 3
      # image_processing.get_big_data(big_scale)

      bit_length = 1
      # bit_length = 2
      # bit_length = 3
      # bit_length = 4
      # image_processing.get_quantized_data(bit_length)

      bit_length = 2
      average = True
      image_processing.get_small_data(bit_length, average)
