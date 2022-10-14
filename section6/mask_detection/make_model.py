import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import cv2, glob
import numpy as np

import warnings
warnings.simplefilter('ignore')


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

#テスト用画像をNumpy形式で得る
x,y=[[],[]]
read_files("./image/mask_off_test/*.jpg", [1,0])
read_files("./image/mask_on_test/*.jpg", [0,1])
x_test, y_test=(np.array(x), np.array(y))


#データの学習
hist=model.fit(
    x_train,
    y_train,
    batch_size=100,
    epochs=100,
    validation_data=(x_test,y_test)
)

#データの評価
score=model.evaluate(x_test, y_test, verbose=1)
print("正解率 = ",score[1], 'loss = ',score[0])

#モデルの保存
model.save('mask_model.h5')

#学習の様子を可視化
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy')
plt.legend(['train','test'], loc='upper left')
plt.show()
