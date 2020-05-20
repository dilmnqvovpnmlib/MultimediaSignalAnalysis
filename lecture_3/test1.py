import cv2
import math
import numpy as np
from matplotlib import pyplot as plt


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

rotate_matrix = lambda radian, x_t, y_t : [
        [math.cos(radian), math.sin(radian), x_t],
        [-math.sin(radian), math.cos(radian), y_t]
    ]

def add_edge(img):
    H, W = img.shape
    WID = 1500# int(np.max(img.shape) * 2 ** 0.5)
    print(WID)
    e_img = np.zeros((WID, WID))
    e_img[
        int((WID-H)/2):int((WID+H)/2),
        int((WID-W)/2):int((WID+W)/2)
    ] = img
    return e_img

def dot_gen(map_matrix):    
    return lambda x: np.dot(map_matrix, x)

def afin_image(afintrans, image, x_length, y_length, ):
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



afintrans = dot_gen(rotate_matrix(0, 220, 42))
path_src = './input/1-004-1.jpg'
inputs_src = cv2.imread(path_src)
edges_src = add_edge(cv2.cvtColor(inputs_src, cv2.COLOR_RGB2GRAY))

path_tmp = './input/1-004-2.jpg'
inputs_tmp = cv2.imread(path_tmp)
edges_tmp = add_edge(cv2.cvtColor(inputs_tmp, cv2.COLOR_RGB2GRAY))

rows,cols = edges_tmp.shape
afin_result = afin_image(afintrans, edges_tmp, cols, rows) + edges_src
plt.subplot(122),plt.imshow(afin_result,cmap = 'gray')
plt.title('Rotate Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
cv2.imwrite("rotate.png", afin_result)