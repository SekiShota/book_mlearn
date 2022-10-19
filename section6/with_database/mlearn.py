"""
100件分のデータを使ってモデルの学習
"""

import tensorflow.keras as keras
from keras.models import load_model
from keras.utils import to_categorical
import numpy as np
import sqlite3
import os

#データベースから最新の100件のデータを取り出す
dbpath="./hw.sqlite3"
select_sql="SELECT * FROM person ORDER BY id DESC LIMIT 10000"

#読み出したデータを元にラベルyとデータxをリストに追加
x=[]
y=[]
with sqlite3.connect(dbpath) as conn:
    for row in conn.execute(select_sql):
        id, height, weight, typeNo=row

        #データを正規化する
        height=height/200
        weight=weight/150
        y.append(typeNo)
        x.append(np.array([height, weight]))

#モデルの読み込み
model=load_model('hw_model.h5')

#すでに学習データがあれば読み込む, 前回の学習結果に加えて新しいデータで学習ができる
if os.path.exists('hw_weights.h5'):
    model.load_weights('hw_weights.h5')

#体型を6段階に分割してone-hot encoding
nb_classes=6
y=to_categorical(y, nb_classes)

#学習
model.fit(np.array(x), y, batch_size=50, epochs=100)

#結果の保存
model.save_weights('hw_weights.h5')
