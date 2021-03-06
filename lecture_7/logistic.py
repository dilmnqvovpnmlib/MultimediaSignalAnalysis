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

#シグモイド関数
def sigmoid(x):
    return 1/(1+np.exp(-x))

#faceファイルの画像読み込み
img_list1 = glob.glob('./input/face/*.jpg')
#nonfaceファイルの画像読み込み
img_list2 = glob.glob('./input/nonface/*.jpg')
random.shuffle(img_list2)#nonface画像のshuffle

#顔画像と非顔画像それぞれのテストデータの個数
n_test=1200

#訓練用とテスト用にわける(jpglist)
train_face_list = read_file(img_list1[:-n_test])
train_nonface_list = read_file(img_list2[:-n_test])
test_face_list = read_file(img_list1[-n_test:])
test_nonface_list = read_file(img_list2[-n_test:])

##データの個数
#L=1000
#M=1100
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
classes, trainlabel_indices = np.unique(trainlabel, return_inverse=True)#face=0,nonface=1のラベルを割り当てる
print(trainlabel_indices)
#訓練データの作成
trainlist= np.concatenate([train_face_list,train_nonface_list])
m = np.mean(trainlist)
std = np.std(trainlist)
#標準化した訓練データを入れるリスト
zscore_trainlist = []
for i in range(len(trainlist)):
    zscore_trainlist.append((trainlist[i] - m) / std)
zscore_trainlist = np.array(zscore_trainlist)

#訓練データの先頭に１を加える処理(np.array) 
Xtil_train = np.c_[np.ones(zscore_trainlist.shape[0]),zscore_trainlist]

#クラスごとに教師ベクトルの割り当て
b_matrix = np.where(trainlabel_indices==0,0,1)#顔画像に0,非顔画像に1を割り当てる

#重みと学習率の初期化
w = np.zeros(Xtil_train.shape[1])
rho=0.00001

#テストデータの前処理
#訓練データの作成
testlist= np.concatenate([test_face_list,test_nonface_list])
#標準化した訓練データを入れるリスト
zscore_testlist = []
for i in range(len(testlist)):
    zscore_testlist.append((testlist[i] - m) / std)
zscore_testlist = np.array(zscore_testlist)
#訓練データの先頭に１を加える処理(np.array) 
Xtil_test = np.c_[np.ones(zscore_testlist.shape[0]),zscore_testlist]

#重みの更新
for i in range(1000):
    w -= rho*(Xtil_train.T @ (sigmoid(Xtil_train @ w) - b_matrix))
    #二乗和誤差を用いて損失関数Eを求める
    E=0
    for j in Xtil_test[:len(test_face_list)]:
        E += (sigmoid(j @ w ) -0)**2 / 2
    for j in Xtil_test[-len(test_nonface_list):]:
        E += (sigmoid(j @ w ) -1)**2 / 2
    print(E)
    if E < 41:#損失関数が41より小さくなれば更新終了
        break
    
#重みの評価
count=0
for j in Xtil_test[:len(test_face_list)]:
    if sigmoid(j @ w ) < 0.5:#出力が0.5以下なら顔画像と判別
        count+=1
for j in Xtil_test[-len(test_nonface_list):]:
    if sigmoid(j @ w ) > 0.5:#出力が0.5以上なら非顔画像と判別
        count+=1
confusion_matrix =(count/len(Xtil_test))*100
print("confusion_matrix:{}".format(confusion_matrix))