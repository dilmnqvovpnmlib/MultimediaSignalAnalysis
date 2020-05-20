import cv2
import math
import numpy as np
from matplotlib import pyplot as plt


def template_matching_ssd(src, temp):
    # 画像の高さ・幅を取得
    h, w = src.shape
    ht, wt = temp.shape[0], temp.shape[1]
    # スコア格納用の二次元配列
    score = np.empty((h-ht, w-wt))
    # テンプレートマッチングの走査
    for dy in range(0, h - ht):
        for dx in range(0, w - wt):
            #　二乗誤差の和を計算
            diff = (src[dy:dy + ht, dx:dx + wt] - temp)**2
            score[dy, dx] = diff.sum()
    
    # スコアが最小の走査位置を返す
    pt = np.unravel_index(score.argmin(), score.shape)

    return (pt[1], pt[0])

# 変換行列を求める関数
rotate_matrix = lambda radian, x_t, y_t : [
        [math.cos(radian), math.sin(radian), x_t],
        [-math.sin(radian), math.cos(radian), y_t]
    ]

# 画像の周りに黒のエッジを加える関数
def add_edge(img):
    H, W = img.shape
    WID = 1500 # int(np.max(img.shape) * 2 ** 0.5)
    print(WID)
    e_img = np.zeros((WID, WID))
    e_img[
        int((WID-H)/2):int((WID+H)/2),
        int((WID-W)/2):int((WID+W)/2)
    ] = img
    return e_img

# 変換行列を作をさせる関数
def dot_gen(map_matrix):    
    return lambda x: np.dot(map_matrix, x)

# アフィン変換をする関数
def afin_image(afintrans, image, x_length, y_length):
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


if __name__ == '__main__':
    # 平行移動させるサイズを求める
    ## 比較対象の2画像をインポート
    path_src = './input/1-004-1.jpg'
    inputs_src = cv2.imread(path_src)
    edges_src = cv2.cvtColor(inputs_src, cv2.COLOR_RGB2GRAY)

    path_comp = './input/test.png'
    inputs_comp = cv2.imread(path_comp)
    edges_comp = cv2.cvtColor(inputs_comp, cv2.COLOR_RGB2GRAY)

    pt = template_matching_ssd(edges_src, edges_comp)
    move_x, move_y = pt[0], pt[1]

    # 平行移動させる画像をインポート
    path_tmp = './input/1-004-2.jpg'
    inputs_tmp = cv2.imread(path_tmp)
    edges_tmp = add_edge(cv2.cvtColor(inputs_tmp, cv2.COLOR_RGB2GRAY))

    # 変換行列を計算
    afintrans = dot_gen(rotate_matrix(0, move_x, move_y))

    rows, cols = edges_tmp.shape

    # アフィン変換
    afin_result = afin_image(afintrans, edges_tmp, cols, rows) + add_edge(cv2.cvtColor(inputs_src, cv2.COLOR_RGB2GRAY))

    cv2.imwrite('./output/ans.png', afin_result)
