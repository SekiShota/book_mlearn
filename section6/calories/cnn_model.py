"""
(1)CNNのアーキテクチャの定義: def_model(画像のサイズ、クラス数)
(2)CNNのコンパイル、構築: get_model(画像のサイズ、クラス数)
→モジュール化して学習を実装するファイルより読み込んで使用する
"""

import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop

"""
# (1)CNNのアーキテクチャの定義: def_model(画像のサイズ、クラス数)
入力層：50,50,3
畳み込み層1: 3x3,32, relu
畳み込み層2: 3x3,32, relu
プーリング層: 2x2,MaxPooling2D
ドロップアウト0.25
---------------------------------
畳み込み層3: 3x3,64, relu
畳み込み層4: 3x3,64, relu
プーリング層: 2x2,MaxPooling2D
ドロップアウト0.25
---------------------------------
平坦化層: Flatten
全結合層: 512, relu
ドロップアウト0.5
出力層: nb_classes, softmax
"""
def def_model(in_shape, nb_classes):
    model=Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=in_shape))
    model.add(Conv2D(32,(3,3),activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.25))

    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    return model

"""
# (2)CNNのコンパイル、構築: get_model(画像のサイズ、クラス数)
損失関数：カテゴリカル交差エントロピー誤差
最適化アルゴリズム：RMSprop()
評価指標：accuracy
"""
def get_model(in_shape, nb_classes):
    model=def_model(in_shape, nb_classes)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=RMSprop(),
        metrics=['accuracy']
    )
    return model
