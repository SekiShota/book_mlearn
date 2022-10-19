"""
任意の身長と体重を入力して、精度の確認をする
"""

import tensorflow.keras as keras
from keras.models import load_model
import numpy as np

#モデルの読み込み
model=load_model('hw_model.h5')

#学習済みデータ(重み)の読み込み
model.load_weights('hw_weights.h5')

#ラベルづけ
LABELS=['低体重','普通体重', '肥満1', '肥満2', '肥満3', '肥満4']

#テストデータの指定
heights=160
weights=50

#データの正規化
test_x=[heights/200, weights/150]

#予測
pred=model.predict(np.array([test_x]))
print(pred)
idx=pred[0].argmax()
print(LABELS[idx], '/可能性：', pred[0][idx])
