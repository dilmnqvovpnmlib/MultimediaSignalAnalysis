import glob

import cv2
import numpy as np
import scipy.spatial.distance as distance
import scipy.stats as stats
from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
from PIL import Image

class knn:
    def __init__(self, k):
        self.k = k
        self._fit_X = None
        self.classes = None
        self._y = None
    
    def fit(self, X, label):
        self._fit_X = X
        self.classes, self.label_indices = np.unique(label, return_inverse=True)

    def neighbors(self, Y):
        dist = distance.cdist(Y, self._fit_X)
        neigh_ind = np.argpartition(dist, self.k)
        neigh_ind = neigh_ind[:, :self.k]
        return neigh_ind

    def predict(self, Y):
        neigh_ind = self.neighbors(Y)
        mode, _ = stats.mode(self.label_indices[neigh_ind], axis=1)
        mode = np.asarray(mode.ravel(), dtype=np.intp)
        result = self.classes.take(mode)
        return result

# データサイズからデータとそのラベルを作成する関数
def getLabelAndData(files, size, type):
    label = [0 if type == 'face' else 1 for _ in range(size)]
    data = []
    for file in files[:size]:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        reshaped_img = img.reshape(-1,)
        data.append(reshaped_img)
    np_data = np.array(data)
    return label, np_data

# テストデータを作成する関数
def getTestData(files):
    data = []
    for file in files:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        reshaped_img = img.reshape(-1,)
        data.append(reshaped_img)
    return data

# 主成分スコアを計算する関数
def getScore(images, u):
    score = []
    for image in images:
        score.append(
            image @ u.T
        )
    return score

# 固有値・固有ベクトルを求めて大基準にソートする関数
def eigsort(S):
    eigv_raw, u_raw = LA.eig(S)
    eigv_index = np.argsort(eigv_raw)[::-1]
    eigv = eigv_raw[eigv_index]
    u = u_raw[:, eigv_index]
    return [eigv, u]

if __name__ == '__main__':
    face_files = glob.glob('./input/face/*.jpg')
    non_face_files = glob.glob('./input/non-face/*.jpg')

    face_data_size = len(face_files)
    non_face_data_size = len(non_face_files)

    for n in [6, 7, 8, 9]:
        print('n', n)
        train_face_data_size = face_data_size * n // 10
        train_non_face_data_size = face_data_size * n // 10

        # test_face_data_size = face_data_size // 5
        # test_non_face_data_size = face_data_size // 5

        # 学習データとそのラベルを作成
        face_labels, np_face_data = getLabelAndData(
            face_files,
            train_face_data_size,
            'face',
        )

        non_face_labels, np_non_face_data = getLabelAndData(
            non_face_files,
            train_non_face_data_size,
            'non_face'
        )

        # テストデータを作成
        face_test_data_size = len(face_files[train_face_data_size:])
        non_test_face_data_size = len(non_face_files[train_non_face_data_size:])
        ans = [0 for _ in range(face_test_data_size)] + [1 for _ in range(non_test_face_data_size)]
        np_test_data = np.array(
            getTestData(face_files[train_face_data_size:])
            + getTestData(non_face_files[train_non_face_data_size:])
        )

        m_np_face_data = np.mean(np_face_data + np_non_face_data, axis=0)

        # 共分散行列を求める
        S = np.zeros((32 * 32, 32 * 32))
        N = train_face_data_size + train_non_face_data_size

        for face in np_face_data:
            tmp = face - m_np_face_data
            S += tmp.reshape(-1, 1) @ tmp.reshape(-1, 1).T

        for face in np_non_face_data:
            tmp = face - m_np_face_data
            S += tmp.reshape(-1, 1) @ tmp.reshape(-1, 1).T
        
        S /= N

        [eig, u_inner] = eigsort(S)
        eig, u_inner = np.abs(eig), np.abs(u_inner)

        y = np.cumsum(eig)
        ylist = y / y[-1]
        # plt.plot([i for i in range(len(ylist))], ylist)
        # plt.show()

        index = np.where(ylist >= 0.8)[0]
        u_list = u_inner[:index[0] + 1]
        print("The number of axes which first achieve 80%:" + str(index[0] +1))
        
        update_np_face_data  = getScore(np_face_data, u_list)
        update_np_non_face_data = getScore(np_non_face_data, u_list)

        k = 4
        print('k', k)
        K = knn(k=k)
        # 元のデータとラベルをセット
        samples = np.concatenate([update_np_face_data, update_np_non_face_data])
        label = np.concatenate([face_labels, non_face_labels])
        K.fit(samples, label)
        # 予測したいデータ
        Y = getScore(np_test_data, u_list)
        p = K.predict(Y)
        target = sum([1 if a == b else 0 for a, b in zip(p, ans)])
        print('total:', len(ans), 'target', target, target / len(ans))
        print('-' * 10)
