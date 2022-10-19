"""
正解データ1000件用意して、モデルで予測した結果の正答率を確認する
"""

import tensorflow.keras as keras
from keras.models import load_model
import numpy as np
import random
from keras.utils import to_categorical

#学習モデル読み込み
model=load_model('hw_model.h5')

#学習済みのデータ（重み）の読み込み
model.load_weights('hw_weights.h5')

#正解データ1000件生成
x=[]
y=[]
for i in range(1000):
    height=random.randint(130, 180)
    weight=random.randint(30, 100)
    #体型データはBMIに基づいて自動生成
    typeNo=1
    bmi=weight/(height/100)**2
    if bmi<18.5:
        typeNo=0
    elif bmi<25:
        typeNo=1
    elif bmi<30:
        typeNo=2
    elif bmi<35:
        typeNo=3
    elif bmi<40:
        typeNo=4
    else:
        typeNo=5

    #データを正規化する
    height=height/200
    weight=weight/150
    x.append(np.array([height, weight]))
    y.append(typeNo)

#形式の変換
x=np.array(x)
y=to_categorical(y, 6)

#正解率の確認
score=model.evaluate(x, y, verbose=1)
print('accuracy: ', score[1], 'loss: ',score[0])
