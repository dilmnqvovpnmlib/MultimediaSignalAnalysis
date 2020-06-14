import numpy as np
import scipy.spatial.distance as distance
import scipy.stats as stats
import cv2

import glob


class knn:
    def __init__(self,k):
        self.k = k
        self._fit_X = None  # 既存データを格納 
        self.classes = None  #
        self._y = None
    def fit(self, X, label):
        # Xは元のデータ点で、shape(data_num, feature_num)
        print("original data:\n", X)
        print("label:\n", label)
        self._fit_X = X
        # ラベルデータからクラスを抽出、またラベルをindexとした配列を作成
        # self.classes[self.label_indices] == label のように復元できるのでreturn_inverseという
        self.classes, self.label_indices = np.unique(label, return_inverse=True)
        print("classes:\n", self.classes)
        print("label_indices:\n", self.label_indices)
        print("classes[label_indices]で復元されるか確認:\n", self.classes[self.label_indices])
    def neighbors(self, Y):
        # Yは予測対象のデータ点(複数可)で, shape(test_num, feature_num) 
        # 予測対象の点とデータ点の距離を求めるので、test_num * data_num だけ距離を計算する
        dist = distance.cdist(Y, self._fit_X)
        print("テストデータと元データとの距離:\n", dist)
        # distはshape(test_num, data_num) となる
        # [[1.41421356 1.11803399 2.6925824  2.23606798]   テスト点1と元データ各点との距離
        #  [3.         2.6925824  1.80277564 1.41421356]   テスト点2と元データ各点との距離
        #  [3.31662479 3.20156212 1.11803399 1.41421356]]  テスト点3と元データ各点との距離

        # 距離を測定したらk番目までに含まれるindexをもとめる
        # argpartitionはk番目までと、それ以降にデータを分ける関数
        # argsortだと距離の順位もわかるが、素のk-nnでは距離順位の情報はいらないので、argpartitionを使う
        neigh_ind = np.argpartition(dist, self.k)
        # neigh_indのshapeは(test_num, feature_num)となる
        # 上のdistでk=2でargpartitionしたときの結果
        # 例えば1行目だと index 2,1 が上位2要素になっている。上の距離をみると、0.5と1.5が相当する
        # 2行目だと index 3, 2 が上位2要素で、1.73と1.80が相当する
        #[[1 0 3 2]
        # [3 2 1 0]
        # [2 3 1 0]]
        # k番目までの情報だけを取り出す
        neigh_ind = neigh_ind[:, :self.k]
        # neigh_indのshapeは(test_num, self.k)となる
        #[[1 0]   テスト点1に近い元データ点のindexのリスト
        # [3 2]   テスト点2に近い元データ点のindexのリスト
        # [2 3]]  テスト点3に近い元データ点のindexのリスト
        return neigh_ind
    def predict(self, Y):
        # k番目までのindexを求める shape(test_num, self.k)となる
        print("test data:\n",Y)
        neigh_ind = self.neighbors(Y)
        # stats.modeでその最頻値を求める. shape(test_num, 1) . _は最頻値のカウント数
        # self.label_indices は [0 0 1 1] で、元データの各点のラベルを表す
        # neigh_indは各テスト点に近い元データのindexのリストで shape(est_num, k)となる
        # self.label_indices[neigh_ind] で、以下のような各テスト点に近いラベルのリストを取得できる
        # [[0 0]  テスト点1に近い元データ点のラベルのリスト
        #  [1 1]  テスト点2に近い元データ点のラベルのリスト
        #  [1 1]] テスト点3に近い元データ点のラベルのリスト
        # 上記データの行方向(axis=1)に対してmode(最頻値)をとり、各テスト点が属するラベルとする
        # _はカウント数
        mode, _ = stats.mode(self.label_indices[neigh_ind], axis=1)
        # modeはaxis=1で集計しているのでshape(test_num, 1)となるので、ravel(=flatten)してやる
        # [[0]
        #  [1]
        #  [1]]
        # なおnp.intpはindexに使うデータ型
        mode = np.asarray(mode.ravel(), dtype=np.intp)
        print("test dataの各ラベルindexの最頻値:\n",mode)
        # index表記からラベル名表記にする. self.classes[mode] と同じ
        result = self.classes.take(mode)
        return result

if __name__ == '__main__':
    face_files = glob.glob('./input/face/*.jpg')
    non_face_files = glob.glob('./input/non-face/*.jpg')

    face_data_size = len(face_files)
    non_face_data_size = len(non_face_files)

    train_face_data_size = face_data_size * 3 // 5
    train_non_face_data_size = face_data_size * 3 // 5

    # test_face_data_size = face_data_size // 5
    # test_non_face_data_size = face_data_size // 5

    face_labels = [0 for _ in range(train_face_data_size)]
    face_data = []
    for face_file in face_files[:train_face_data_size]:
        img = cv2.imread(face_file, cv2.IMREAD_GRAYSCALE)
        reshaped_img = img.reshape(-1,)
        face_data.append(reshaped_img)
        print(reshaped_img, reshaped_img.shape[0])
    np_face_data = np.array(face_data)

    non_face_labels = [1 for _ in range(train_non_face_data_size)] 
    non_face_data = []  
    for non_face_file in non_face_files[:train_non_face_data_size]:
        img = cv2.imread(non_face_file, cv2.IMREAD_GRAYSCALE)
        reshaped_img = img.reshape(-1,)
        non_face_data.append(reshaped_img)
        print(reshaped_img, reshaped_img.shape[0])
    np_non_face_data = np.array(non_face_data)

    ans = [0 for _ in range(len(face_files[train_face_data_size:]))] + [0 for _ in range(len(non_face_files[train_non_face_data_size:]))]
    test_data = []
    for face_file in face_files[train_face_data_size:]:
        img = cv2.imread(face_file, cv2.IMREAD_GRAYSCALE)
        reshaped_img = img.reshape(-1,)
        test_data.append(reshaped_img)
        print(reshaped_img, reshaped_img.shape[0])
    for non_face_file in non_face_files[train_non_face_data_size:]:
        img = cv2.imread(non_face_file, cv2.IMREAD_GRAYSCALE)
        reshaped_img = img.reshape(-1,)
        test_data.append(reshaped_img)
        print(reshaped_img, reshaped_img.shape[0])
    np_test_data = np.array(test_data)

    k = 2
    K = knn(k=k)
    # 元のデータとラベルをセット
    samples = np.concatenate([np_face_data, np_non_face_data]) # [[0., 0., 0.], [0., .5, 0.], [1., 2., -2.5],[1., 2., -2.]]
    label = np.concatenate([face_labels, non_face_labels]) # ['a','a','b', 'b']
    K.fit(samples, label)
    # 予測したいデータ
    Y = np_test_data # [[1., 1., 0.],[2, 2, -1],[1, 1, -3]]
    p = K.predict(Y)
    print("result:\n", p, len(p))
    print('total:', len(ans))
    target = 0
    for a, b in zip(p, ans):
        if a == b:
            target += 1
    print(target, target / len(ans))