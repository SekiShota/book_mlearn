import tensorflow.keras as keras
from keras.models import load_model
import cv2, dlib, pprint, os
import numpy as np
import pygame.mixer
import time

"""
マスクをつけていない時にアラームを鳴らす関数
"""
def sound(f,i,n):
    pygame.mixer.init() #初期化します
    pygame.mixer.music.load(f) #音声ファイルを読み込みます
    pygame.mixer.music.play(i) #再生します
    time.sleep(n) #再生時間を指定します
    pygame.mixer.music.stop() #終了します

#結果ラベル
res_labels=['NO MASK!', 'OK']
save_dir="./live"

#保存した学習モデルを読み込む
model=load_model('mask_model.h5')

#dlibを始める
detector=dlib.get_frontal_face_detector()

#webカメラから入力を開始
red=(0,0,255)
green=(0,255,0)
fid=1
capture=cv2.VideoCapture(1)

while True:
    #カメラの画像を読み込む
    ok, frame=capture.read()
    if not ok:
        break

    #画面を縮小表示する
    frame=cv2.resize(frame, (800,600))

    #顔検出
    dets=detector(frame, 1)
    for k,d in enumerate(dets):
        pprint.pprint(d)
        x1=int(d.left())
        y1=int(d.top())
        x2=int(d.right())
        y2=int(d.bottom())

        #顔部分を切り取る
        im=frame[y1:y2, x1:x2]
        im=cv2.resize(im, (50,50))
        im=im.reshape(-1, 50, 50, 3)

        #予測
        res=model.predict([im])[0]
        v=res.argmax()
        print(res_labels[v])

        #枠を描画, マスクない時(v=0)は赤で強調する,
        #マスクあり：サイレンを鳴らします
        #マスクなし：サイレンを鳴らします
        # color=green if v==1 else red
        # border=2 if v==1 else 5
        if v==1:
            color=green
            border=2
            sound("Quiz-Correct_Answer02-1.mp3",2,3)
        else:
            color=red
            border=10
            sound("Warning-Siren05-02(Fast-Long).mp3",2,5)

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness=border)

        #テキストを描画
        cv2.putText(frame, res_labels[v], (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, thickness=2)

    #結果を保存
    if len(dets)>0:
        if os.path.exists(save_dir):
            jpgfile=save_dir+"/"+str(fid)+".jpg"
            cv2.imwrite(jpgfile, frame)
            fid+=1

    #ウィンドウに画像を出力
    cv2.imshow('Mask Live Check', frame)

    #ESC or Enterキーでループ脱出
    k=cv2.waitKey(1)
    if k==13 or k==27:
        break

#カメラを開放
capture.release()
#ウィンドウを破棄
cv2.destroyAllWindows()
