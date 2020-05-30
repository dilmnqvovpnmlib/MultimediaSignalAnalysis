import math

import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.random import randint
from PIL import Image, ImageDraw


def k_mean(original_img, k):
    #STEP1
    w, h = original_img.size

    # 画像を扱いやすい２次配列に変換
    img_pixels = np.array([[img.getpixel((x,y)) for x in range(w)] for y in range(h)])
    # 減色画像用の２次配列も用意しておく
    reduce_img_pixels = np.array([[(0, 0, 0) for x in range(w)] for y in range(h)])

    # 代表色の初期値をランダムに設定
    class_values = []
    for i in range(k):
        class_values.append(np.array([randint(256), randint(256), randint(256)]))

    #STEP2
    # 20回繰り返す
    for i in range(10):
        #STEP2-1
        print("ranning at iteration No." + str(i))
        sums = []
        for i in range(k):
            sums.append(np.array([0, 0, 0])) 
            sums_count = [0] * k

        #STEP2-2
        # 各画素のクラスを計算
        for x in range(w):
            for y in range(h):
                min_d = (256 ** 2) * 3
                class_index = 0
                # 一番近い色（クラス）を探索
                for j in range(k):
                    d = sum([x*x for x in img_pixels[y][x] - class_values[j]])
                    if min_d > d:
                        min_d = d
                        class_index = j
                sums[class_index] += img_pixels[y][x]
                sums_count[class_index] += 1
                reduce_img_pixels[y][x] = tuple(list(map(int, class_values[class_index])))

        #STEP2-3
        # 代表色を更新
        for m in range(k):
            class_values[m] = sums[m] / sums_count[m]

    # STEP3
    # ２次元配列から加工後の画像へ変換
    reduce_img = Image.new('RGB', (w, h))
    for x in range(w):
        for y in range(h):
            reduce_img.putpixel((x, y), tuple(reduce_img_pixels[y][x]))

    return reduce_img

if __name__ == '__main__':
    # im = Image.open("./input/1-009-1.jpg")
    # k平均法による減色処理

    # 画像ファイルの読み込み
    filename = "./input/test.jpg"
    img = Image.open(filename).convert("RGB")

    # k平均法による減色処理
    reduce_img = k_mean(img, 5)

    # 画像データの更新とファイル出力
    reduce_img.save("reduce_image_test.jpg")
