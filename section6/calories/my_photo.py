"""
画像を読み込んで予測してみる
"""

import cnn_model
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#ファイル名指定
target_image1="./image/test-sushi.jpg"
target_image2="./image/test-sarad.jpg"


#画像サイズ指定
im_rows=32
im_cols=32
im_color=3
in_shape=(im_rows, im_cols, im_color)
nb_classes=3

#ラベルとカロリーを指定
LABELS=['お寿司', 'サラダ', '麻婆豆腐']
CALORIES=[588, 118, 648]

#保存したCNNモデルの読み込み
clf=cnn_model.get_model(in_shape, nb_classes)
clf.load_weights('./image/photos-model-light.hdf5')

#画像を読み込んで、予測する
def check_photo(path):
    img=Image.open(path)
    img=img.convert("RGB")
    img=img.resize((im_cols, im_rows))
    plt.imshow(img)
    plt.show()
    #データ変換
    x=np.asarray(x)
    x=x.reshape(-1, im_rows, im_cols, im_color)
    x=x/255

    #予測
    pred=clf.predict([x])[0]
    idx=pred.argmax()
    per=int(pred[idx]*100)
    return (idx, per)

#結果を表示
def check_photo_str(path):
    idx, per=check_photo(path)
    #答えを表示
    print("この写真は", LABELS[idx], "で、カロリーは", CALORIES[idx], "kcal")
    print("可能性は", per, "%")

if __name__=='___main__':
    check_photo_str(target_image1)
    check_photo_str(target_image2)
