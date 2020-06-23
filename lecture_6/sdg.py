import glob

import numpy as np

from modules import  getLabelAndData, getTestData


face_files = glob.glob('./input/face/*.jpg')
non_face_files = glob.glob('./input/non-face/*.jpg')

face_data_size = len(face_files)
non_face_data_size = len(non_face_files)

n = 1
train_face_data_size = 1000#face_data_size * n // 10
train_non_face_data_size = 1000# non_face_data_size * n // 10

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

ans = [1 for _ in range(1000)] + [-1 for _ in range(1000)]
np_test_data = np.array(
    getTestData(face_files[train_face_data_size:train_face_data_size+1000])
    + getTestData(non_face_files[train_non_face_data_size:train_non_face_data_size+1000])
)

rho = 10 ** (-3)

# b(教師ベクトル) を作成
face = np.array([-1 for _ in range(train_face_data_size)])
face[0] = 1
non_face = np.array([-1 for _ in range(train_non_face_data_size)])
non_face[1] = 1

# W(重みベクトル) を作成
w_face = np.random.normal(loc=0, scale=0.01, size=1025)
w_non_face = np.random.normal(loc=0, scale=0.01, size=1025)

for i in range(2000):
    if i < 1000:
        w_face = w_face - rho * (w_face.reshape(-1,) @ samples[i] - 1) * samples[i]
    else:
        w_non_face = w_non_face - rho * (w_non_face.reshape(-1,) @ samples[i] - 1) * samples[i]


# 正答率を計算
N = np_test_data.shape[0]
np_test_data = np.hstack([np.array([np.ones(N)]).T, np_test_data])
face_ans = np_test_data @ w_face
non_face_ans = np_test_data @ w_non_face

count = 0
print(len(ans), len(face_ans), len(non_face_ans))
for i, (a, b) in enumerate(zip(face_ans, non_face_ans)):
    value = 1 if a >= b else -1
    count += 1 if value == ans[i] else 0
print('正答率', count / len(ans))
