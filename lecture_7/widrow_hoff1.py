# 必要なものをimport
from PIL import Image
import numpy as np
import glob
import random

# 画像の読み込みかつ32*32の行列の一次元配列化
def read_file(Imagelist):
    data=[]
    for file in Imagelist:
        img = np.array( Image.open(file).convert("L"),'f' )
        data.append(img.flatten())#全ての画素数を一次元化して格納
    return data

#faceファイルの画像読み込み
img_list1 = glob.glob('face/*.jpg')
#nonfaceファイルの画像読み込み
img_list2 = glob.glob('nonface/*.jpg')
random.shuffle(img_list2)#nonface画像のshuffle

#テストデータの個数
n_test=1200

#訓練用とテスト用にわける(jpglist)
train_face_list = read_file(img_list1[:-n_test])
train_nonface_list = read_file(img_list2[:-n_test])
test_face_list = read_file(img_list1[-n_test:])
test_nonface_list = read_file(img_list2[-n_test:])

##データの個数
#L=1000
#M=1500
#
##訓練用とテスト用にわける(jpglist)
#train_face_list = read_file(img_list1[:L])
#train_nonface_list = read_file(img_list2[:L])
#test_face_list = read_file(img_list1[L:M])
#test_nonface_list = read_file(img_list2[L:M])

#訓練データのラベル作成
face_label=['face']
nonface_label=['nonface']
trainlabel = face_label*len(train_face_list) + nonface_label*len(train_nonface_list)
classes2, trainlabel_indices = np.unique(trainlabel, return_inverse=True)#face=0,nonface=1のラベルを割り当てる


#訓練データの作成
trainlist= np.concatenate([train_face_list,train_nonface_list])
m = np.mean(trainlist)
std = np.std(trainlist)
zscore_trainlist = []#標準化した訓練データを入れるリスト
for i in range(len(trainlist)):
    zscore_trainlist.append((trainlist[i] - m) / std)
zscore_trainlist = np.array(zscore_trainlist)

#訓練データの先頭に１を加える処理(np.array) 
Xtil_train = np.c_[np.ones(zscore_trainlist.shape[0]),zscore_trainlist]


#クラスごとに教師ベクトルの割り当て
b_face_matrix = np.where(trainlabel_indices==0,1,-1)
b_nonface_matrix = np.where(trainlabel_indices==0,-1,1)
print(b_face_matrix)
#重みの更新
w_face = np.linalg.inv(Xtil_train.T @ Xtil_train) @ Xtil_train.T @ b_face_matrix.T
w_nonface = np.linalg.inv(Xtil_train.T @ Xtil_train) @ Xtil_train.T @ b_nonface_matrix.T
print(w_face[:10],  w_nonface[:10])
print(w_nonface.T @ Xtil_train[1])

#テストデータの前処理
    #訓練データの作成
testlist= np.concatenate([test_face_list,test_nonface_list])
zscore_testlist = []#標準化した訓練データを入れるリスト
for i in range(len(testlist)):
    zscore_testlist.append((testlist[i] - m) / std)
zscore_testlist = np.array(zscore_testlist)

#訓練データの先頭に１を加える処理(np.array) 
Xtil_test = np.c_[np.ones(zscore_testlist.shape[0]),zscore_testlist]
print(w_face.T @ Xtil_test[1])
print(w_nonface.T @ Xtil_test[1])

#更新した重みを用いて顔/非顔判別
count=0
for i in Xtil_test[:len(test_face_list)]:
    if abs((w_face.T @ i ) -1)< abs((w_nonface.T @ i) -1):#
        count+=1
for i in Xtil_test[-len(test_nonface_list):]:
    if abs((w_face.T @ i )-1) > abs((w_nonface.T @ i)-1) :
        count+=1
confusion_matrix =(count/len(Xtil_test))*100
print("confusion_matrix:{}".format(confusion_matrix))