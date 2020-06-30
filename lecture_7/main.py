import glob

import numpy as np

from modules import  getLabelAndData, getTestData


sigmoid = lambda x: 1 / (1 + np.exp(-x))

face_files = glob.glob('./input/face/*.jpg')
non_face_files = glob.glob('./input/non-face/*.jpg')

face_data_size = len(face_files)
non_face_data_size = len(non_face_files)

n = 1
train_face_data_size = 4500#face_data_size * n // 10
train_non_face_data_size = 4500# non_face_data_size * n // 10

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

# 元のデータとラベルをセット
samples = np.concatenate([np_face_data, np_non_face_data])
label = np.concatenate([face_labels, non_face_labels])

# 先頭に1を挿入
length = samples.shape[0]
samples = np.hstack([np.array([np.ones(length)]).T, samples])

# テストデータを作成
face_test_data_size = len(face_files[train_face_data_size:])
non_test_face_data_size = len(non_face_files[train_non_face_data_size:])

ans_length = 1000
ans = [1 for _ in range(ans_length)] + [-1 for _ in range(ans_length)]
np_test_data = np.array(
    getTestData(face_files[train_face_data_size:train_face_data_size+ans_length])
    + getTestData(non_face_files[train_non_face_data_size:train_non_face_data_size+ans_length])
)

rho = 10 ** (-4)

# b(教師ベクトル) を作成
face = np.array([-1 for _ in range(train_face_data_size)])
face[0] = 1
non_face = np.array([-1 for _ in range(train_non_face_data_size)])
non_face[1] = 1
b = np.concatenate([face, non_face])

# W(重みベクトル) を作成
w_face = np.random.normal(loc=0, scale=0.01, size=1025)
w_non_face = np.random.normal(loc=0, scale=0.01, size=1025)
w = np.random.normal(loc=0, scale=0.01, size=1025)

loop = 1000
for i in range(loop):
    a = samples @ w_face
    tmp = sigmoid(a) - b
    caled = samples.T @ tmp
    w -= rho * caled

# 正答率を計算
N = np_test_data.shape[0]
np_test_data = np.hstack([np.array([np.ones(N)]).T, np_test_data])
face_ans = np_test_data @ w
non_face_ans = np_test_data @ w

count = 0
print(len(ans), len(face_ans), len(non_face_ans))
for index, item in enumerate(face_ans):
    if index < ans_length:
        count += 1 if sigmoid(item) < 0.5 else 0
    if index >= ans_length:
        count += 1 if sigmoid(item) > 0.5 else 0
print(count)
print('正答率', count / len(ans))
