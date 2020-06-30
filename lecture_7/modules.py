import random

import cv2
import numpy as np
import scipy
from sklearn.datasets import make_blobs


# データサイズからデータとそのラベルを作成する関数
def getLabelAndData(files, size, type):
    label = [1 if type == 'face' else -1 for _ in range(size)]
    data = []
    for file in files[:size]:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        reshaped_img = img.reshape(-1,)
        # 平坦化された画像データを正規化
        data.append(scipy.stats.zscore(reshaped_img))
    # 画像をシャッフル
    random.shuffle(data)
    np_data = np.array(data)
    return label, np_data

# テストデータを作成する関数
def getTestData(files):
    data = []
    for file in files:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        reshaped_img = img.reshape(-1,)
        # 平坦化された画像データを正規化
        data.append(scipy.stats.zscore((reshaped_img)))
    # 画像をシャッフル
    random.shuffle(data)
    return data
