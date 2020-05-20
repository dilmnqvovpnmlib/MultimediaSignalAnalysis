import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class ImageProcessing:
    def __init__(self, path, number):
        self.path = path
        self.number = number
        self.im, self.rgb = self.get_image_data(path)
        self.width, self.height = self.im.size
        self.kernel = np.array([
            [1/16, 2/16, 1/16],
            [2/16, 4/16, 2/16],
            [1/16, 2/16, 1/16],
        ])
        # self.show(self.rgb, 'normal', 255)

    def get_image_data(self, path):
        im = Image.open(path)
        rgb = np.array(im)
        return im, rgb

    def show(self, data, type, scale):
        pilImg = Image.fromarray(np.uint8(data))
        pilImg.show()
        # pilImg.save('./{0}/{0}_{1}.jpg'.format(type, scale))
        return pilImg

    def get_monochrome_data(self):
        monochrome_data = 0.299 * self.rgb[:, :, 0] + 0.587 * self.rgb[:, :, 1] + 0.114 * self.rgb[:, :, 2]
        array = []
        for i in range(len(monochrome_data)):
            array += [*monochrome_data[i]]
        array = np.array(array)
        return array

    def get_contrast_data(self):
        is_show = True

        array = self.get_monochrome_data()

        if is_show:
            plt.hist(array, range(0, 255))
            plt.xlabel('pixel values')
            plt.ylabel('ratio of pixels')
            plt.savefig('./data/contrast/image_{}_of_hist.jpg'.format(self.number))
            plt.show()

        max_value, min_value = max(array), min(array)
        contrast_array = (array - min_value) / (max_value - min_value) * 255

        if is_show:
            plt.hist(contrast_array, range(0, 255))
            plt.xlabel('pixel values')
            plt.ylabel('ratio of pixels')
            plt.savefig('./data/contrast/image_{}_of_hist_of_no.jpg'.format(self.number))
            plt.show()
        cv2.imwrite('./data/contrast/image_{}.jpg'.format(self.number), contrast_array.reshape(self.height, self.width))

    def get_cdf_data(self):
        is_show = True

        img = np.array(Image.open(self.path).convert('L'), 'f')
        hist, bins = np.histogram(img.flatten(), bins=256)

        # if is_show:
        #     plt.plot(hist)
        #     plt.xlim(0, 256)
        #     plt.show()

        cdf = hist.cumsum()
        cdf = 255 * cdf / cdf[-1]

        if is_show:
            plt.plot(cdf)
            plt.xlim(0, 256)
            plt.ylim(0, 256)
            plt.show()

        # 線形補間
        img2 = np.interp(img.flatten(), bins[:-1], cdf)
        hist2, bins2 = np.histogram(img2, bins=255)

        if is_show:
            plt.plot(hist2)
            plt.xlim(0, 256)
            plt.xlabel('pixel values')
            plt.ylabel('ratio of pixels')
            plt.savefig('./data/contrast/image_{}_of_hist_of_cdf.jpg'.format(self.number))
            plt.show()
        cv2.imwrite('./data/contrast/image_{}_cdf.jpg'.format(self.number), img2.reshape(img.shape))

    def display_cv_image(self, img, e_img, format='.jpg'):
        cv2.imwrite('./data/contrast/test.jpg', img)

    def add_edge(self):
        img = cv2.imread(self.path, 0)
        H, W = img.shape
        WID = int(np.max(img.shape) * 2 ** 0.5)
        e_img = np.zeros((WID, WID))
        e_img[
            int((WID-H)/2):int((WID+H)/2),
            int((WID-W)/2):int((WID+W)/2)
        ] = img
        self.display_cv_image(img, e_img, '.jpg')
        return e_img

    def affin(self, img, m):
        WID = np.max(img.shape)
        x = np.tile(np.linspace(-1, 1, WID).reshape(1, -1), (WID, 1))
        y = np.tile(np.linspace(-1, 1, WID).reshape(-1, 1), (1, WID))
        p = np.array([[x, y, np.ones(x.shape)]])
        dx, dy, _ = np.sum(p * m.reshape(*m.shape, 1, 1), axis=1)
        u = np.clip((dx + 1) * WID / 2, 0, WID-1).astype('i')
        v = np.clip((dy + 1) * WID / 2, 0, WID-1).astype('i')
        return img[v, u]

    def rotation_matrix(self, data):
        return np.array([
            [np.cos(data), -np.sin(data), 0],
            [np.sin(data),  np.cos(data), 0],
            [0, 0, 1]
            ])

    def geometric_transformation(self):
        e_img = self.add_edge()
        m = self.rotation_matrix(np.pi / 4)
        self.display_cv_image(self.affin(e_img, m), '.jpg')
    
    def local_processing(self, fill_value=-1):
        # get kernel size
        # カーネルサイズ
        src = cv2.imread(self.path, 0)
        m, n = self.kernel.shape

        # width of skip
        # 畳み込み演算をしない領域の幅
        d = int((m-1)/2)

        # get width height of input image
        # 入力画像の高さと幅
        h, w = src.shape[0], src.shape[1]

        # ndarray of destination
        # 出力画像用の配列
        if fill_value == -1:
            dst = src.copy()
        elif fill_value == 0:
            dst = np.zeros((h, w))
        else:
            dst = np.zeros((h, w))
            dst.fill(fill_value)

        # Spatial filtering
        # 畳み込み演算
        for y in range(d, h - d):
            for x in range(d, w - d):
                dst[y][x] = np.sum(src[y-d:y+d+1, x-d:x+d+1] * self.kernel)
        cv2.imwrite('./data/contrast/local.jpg', dst)
        return dst

    # def local_processing(self):
    #     pass


if __name__ == '__main__':
    path_1 = './input/1-0004-1.jpg'
    path_2 = './input/1-0004-2.jpg'
