"""
データとモデルを読み込んで学習する
データは、角度をつけたもの、反転したものを加える
予測精度の確認
"""
import cnn_model
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2

#入力と出力の指定
#画像の縦、横、色空間
im_rows=32
im_cols=32
im_color=3
in_shape=(im_rows, im_cols, im_color)
nb_classes=3

#画像データの読み込み
images=np.load('./image/photos.npz')
x=images['x']
y=images['y']

#読み込んだデータを三次元に変換
x=x.reshape(-1, im_rows, im_cols, im_color)
x=x.astype('float32')/255

#ラベルデータをone-hot encoding, to_categorical
y=keras.utils.to_categorical(y.astype('int32'), nb_classes)

#学習用とテスト用に分割
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=1)

"""
#学習用データの水増し
"""
x_new=[]
y_new=[]
for i, xi in enumerate(x_train):
    yi=y_train[i]
    for ang in range(-30, 30, 5):
        #回転
        center=(16, 16)
        mtx=cv2.getRotationMatrix2D(center, ang, 1.0)
        xi2=cv2.warpAffine(xi, mtx, (32,32))
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

#CNNモデルの取得
clf=cnn_model.get_model(in_shape, nb_classes)

#学習の実行
hist=clf.fit(x_train, y_train, batch_size=32, epochs=20, verbose=1, validation_data=(x_test,y_test))

#モデルの評価
score=clf.evaluate(x_test, y_test, verbose=1)
print(f'accuracy: {score[1]}/loss: {score[0]}')

#学習の様子を可視化
#正解率
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#損失
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

clf.save_weights('./image/photos-model-light.hdf5')

#accuracy: 0.8999999761581421/loss: 1.8037749528884888
