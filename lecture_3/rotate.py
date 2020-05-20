import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from IPython.display import display, Image

class ImageProcessing:
    def __init__(self, path_src, path_tmp):
        self.img = cv2.imread(path_src)
        temp = cv2.imread(path_tmp)

        # グレースケール変換
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        self.temp = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)

        # テンプレート画像の高さ・幅
        self.h_img, self.w_img = self.img.shape[0], self.img.shape[1]
        self.h, self.w = self.temp.shape


    def template_matching_ssd(self, src, temp):
        # 画像の高さ・幅を取得
        h, w = src.shape
        ht, wt = temp.shape[0], temp.shape[1]
        print(h, w)
        print(temp.shape, ht, wt)

        # スコア格納用の二次元配列
        score = np.empty((h-ht, w-wt))

        # 走査
        for dy in range(0, h - ht):
            for dx in range(0, w - wt):
                #　二乗誤差の和を計算
                diff = (src[dy:dy + ht, dx:dx + wt] - temp)**2
                score[dy, dx] = diff.sum()
        
        print(score)

        # スコアが最小の走査位置を返す
        pt = np.unravel_index(score.argmin(), score.shape)

        return (pt[1], pt[0])
    
    def display_cv_image(self, img, e_img, format='.jpg'):
        cv2.imwrite('./output/test.png', img)

    def translation_matrix(self, tx, ty):
        return np.array(
            [
                [1, 0, -tx],
                [0, 1, -ty],
                [0, 0, 1]
            ]
        )

    def add_edge(self, img):
        H, W = img.shape
        WID = int(np.max(img.shape) * 2 ** 0.5)
        e_img = np.zeros((WID, WID))
        e_img[
            int((WID-H)/2):int((WID+H)/2),
            int((WID-W)/2):int((WID+W)/2)
        ] = img
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

    def afin_image(self, afintrans, image, x_length, y_length, ):
        afin_result = np.zeros((y_length, x_length))
        for y in range(len(image)):
            for x in range(len(image[y])):
                if image[y][x] == 0:
                    continue
                trans_posi = afintrans([[x],[y],[1]])
                new_x = math.floor(trans_posi[0][0])
                new_y = math.floor(trans_posi[1][0])
                if new_x >= 0 and new_x < x_length and new_y >= 0 and new_y < y_length:
                    afin_result[new_y][new_x] = image[y][x]
        return afin_result
    
    def main(self):
        # e_img = self.add_edge(self.temp)
        # x, y = 1, 1#220, 42# pt
        # m = self.translation_matrix(x, y)
        # # self.display_cv_image(self.affin(e_img, m), '.jpg')
        # # pt = self.template_matching_ssd(self.gray, self.temp)
        # # x, y = 0, 1# 220, 42# pt
        # self.display_cv_image(self.affin(self.temp, m), '.png')


        # # テンプレートマッチングの結果を出力
        pt = self.template_matching_ssd(self.gray, self.temp)
        print(pt)
        cv2.rectangle(self.img, (pt[0], pt[1]), (pt[0] + self.w, pt[1] + self.h), (0, 0, 200), 3)
        # # 結果を出力
        cv2.imwrite('./output/ssd2.png', self.img)



if __name__ == '__main__':
    path_src = './input/2-004-1.jpg'
    path_tmp = './input/tmp_3.png'

    process = ImageProcessing(path_src, path_tmp)
    process.main()