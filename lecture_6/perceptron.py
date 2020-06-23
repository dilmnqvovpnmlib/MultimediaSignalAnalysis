import glob

import numpy as np

from modules import getLabelAndData, getTestData


# ステップ関数の実装
def Phi(w, x):
    phi_x = w.T @ x
    if phi_x > 0:
        disc = 1
    else:
        disc = -1
    return [phi_x, disc]


# 重みを更新する関数の実装
def OptimalWeight(w, x, t, c):
    phi_x, disc = Phi(w, x)
    if disc * t > 0:
        w_opt = w
    else:
        w_opt = w + (c * t * x)
    return [w_opt, disc*t]


# ファイル名を再帰的に読み込み
face_files = glob.glob('./input/face/*.jpg')
non_face_files = glob.glob('./input/non-face/*.jpg')

face_data_size = len(face_files)
non_face_data_size = len(non_face_files)

n = 8
train_face_data_size = face_data_size * n // 10
train_non_face_data_size = non_face_data_size * n // 10

print("教師データ数: {}".format(train_face_data_size + train_non_face_data_size))

# 学習データ(顔画像)とそのラベルを作成
face_labels, np_face_data = getLabelAndData(
    face_files,
    train_face_data_size,
    'face',
)

# 学習データ(非顔画像)とそのラベルを作成
non_face_labels, np_non_face_data = getLabelAndData(
    non_face_files,
    train_non_face_data_size,
    'non_face'
)

# 元のデータとそのラベルを作成
samples = np.concatenate([np_face_data, np_non_face_data])
label = np.concatenate([face_labels, non_face_labels])

# テストデータを作成
face_test_data_size = len(face_files[train_face_data_size:])
non_test_face_data_size = len(non_face_files[train_non_face_data_size:])

ans = [1 for _ in range(face_test_data_size)] + [-1 for _ in range(non_test_face_data_size)]
np_test_data = np.array(
    getTestData(face_files[train_face_data_size:])
    + getTestData(non_face_files[train_non_face_data_size:])
)

# 教師データ
x_train, t_train = samples, label
N = x_train.shape[0]
D = x_train.shape[1]

# 先頭に1を挿入
ones = np.ones(N)
x_train = np.hstack([np.array([ones]).T, x_train])

N = x_train.shape[0]
D = x_train.shape[1]

w_init = np.zeros(D).T
w_opt = w_init
Disc = np.zeros(N)

# 重みを更新する処理
iter = 1500
for i in range(iter):
    for n in range(N):
        w_opt, Disc[n] = OptimalWeight(w_opt, x_train[n, :], t_train[n], c=0.5)
    if all([e > 0 for e in Disc]):
        break

N = np_test_data.shape[0]
D = np_test_data.shape[1]

# 先頭に1を挿入
ones = np.ones(N)
np_test_data = np.hstack([np.array([ones]).T, np_test_data])
length = np_test_data.shape[0]

print("テスト数: {}".format(length))

# 重みとテストデータを用いて検証
count = 0
for i in range(length):
    data = w_opt.T @ np_test_data[i,:]
    value = 1 if data > 0 else -1
    count += 1 if value == ans[i] else 0

print('epoc', iter, )
print('-' * 10)
print("正答率", count / length)
print('-' * 10)
