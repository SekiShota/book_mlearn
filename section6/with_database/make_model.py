"""
身長、体重、体型を学習
ここではモデル構造の定義のみ
"""

import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

#身長と体重の二次元
in_size=2
#体型は6段階
nb_classes=6

#MLPモデル構造の定義
#入力層；２
#隠れ層：512, 活性化関数：relu
#出力層：6, 活性化関数：softmax
model=Sequential()
model.add(Dense(512, activation='relu', input_shape=(in_size,)))
model.add(Dropout(0.25))
model.add(Dense(nb_classes, activation='softmax'))

#モデルのコンパイル(構築)
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(),
    metrics=['accuracy']
)

#モデルの保存
model.save('hw_model.h5')
print('model was architected and saved')
