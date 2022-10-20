import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import cv2, glob
import numpy as np
import time

import warnings
warnings.simplefilter('ignore')

start=time.time()
print("modules imported!>>>>>>>>>>>>>>>>>>>>>>")

#画像形式の指定
in_shape=(50, 50, 3)
nb_classes=2

#CNNモデル構造の定義
#入力層：50x50x3ch
#畳み込み層1: 3x3のカーネルを32個使う
#畳み込み層2: 3x3のカーネルを32個使う
#プーリング層1: 2x2で区切ってその中の最大値を使う
model=Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=in_shape))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#畳み込み層3: 3x3のカーネルを64個使う
#畳み込み層4: 3x3のカーネルを64個使う
#プーリング層1: 2x2で区切ってその中の最大値を使う
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#全結合層: 512
#出力層: 2(マスクありorなしの2値)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))


#モデルのコンパイル
model.compile(
    loss='categorical_crossentropy',
    optimizer=RMSprop(),
    metrics=['accuracy']
)
print('model compiled!>>>>>>>>>>>>>>>>>>>>')

#画像データをNumpy形式に変換
x=[]
y=[]
def read_files(target_files, y_val):
    files=glob.glob(target_files)
    for fname in files:
        # print(fname)
        #画像の読み出し
        img=cv2.imread(fname)
        #画像サイズを50 x 50にリサイズ
        img=cv2.resize(img, (50,50))
        # print(img)
        x.append(img)
        y.append(np.array(y_val))

#ディレクトリ内の画像を集める
read_files("./image/mask_off/*.jpg", [1,0])
read_files("./image/mask_on/*.jpg", [0,1])
x_train, y_train=(np.array(x), np.array(y))
#
# """
# #学習用データの水増し
# """
x_new=[]
y_new=[]
for i, xi in enumerate(x_train):
    yi=y_train[i]
    for ang in range(-30, 30, 5):
        #回転
        center=(25, 25)
        mtx=cv2.getRotationMatrix2D(center, ang, 1.0)
        xi2=cv2.warpAffine(xi, mtx, (50,50))
        x_new.append(xi2)
        y_new.append(yi)

        #左右反転
        xi3=cv2.flip(xi2,1)
        x_new.append(xi3)
        y_new.append(yi)

#水増しした画像を学習用に置き換える
print("水増し前：", len(y_train))
x_train=np.array(x_new)
y_train=np.array(y_new)
print("水増し後：", len(y_train))

#テスト用画像をNumpy形式で得る
x,y=[[],[]]
read_files("./image/mask_off_test/*.jpg", [1,0])
read_files("./image/mask_on_test/*.jpg", [0,1])
x_test, y_test=(np.array(x), np.array(y))

#早期終了を加える
es_cb = EarlyStopping(patience = 10, restore_best_weights = True)

#データの学習
hist=model.fit(
    x_train,
    y_train,
    batch_size=100,
    epochs=100,
    validation_split=0.2,
    callbacks=[es_cb],
    validation_data=(x_test,y_test)
)
end=time.time()

#データの評価
score=model.evaluate(x_test, y_test, verbose=1)
print("正解率 = ",score[1], 'loss = ',score[0])
print("実行時間：", end-start)

#モデルの保存
model.save('mask_model.h5')

#学習の様子を可視化
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy')
plt.legend(['train','test'], loc='upper left')
plt.savefig('CNN_default_acc.png')
plt.show()


"""
データの水増し前の精度（早期終了なし：100）
正解率 =  0.9888888597488403 loss =  0.0822678655385971
実行時間： 313.0049030780792

データ水増し前の精度（早期終了あり：40）
正解率 =  0.9888888597488403 loss =  0.024253712967038155
実行時間： 128.67297220230103

データ水増し後の精度（早期終了あり：12）
正解率 =  0.9777777791023254 loss =  0.1698571741580963
実行時間： 870.7844309806824

"""
