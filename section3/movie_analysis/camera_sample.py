#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np

#webカメラから入力開始
cap=cv2.VideoCapture(0)

while True:
    #カメラから画像を読み込む
    _, frame=cap.read()
    
    #画像を縮小表示する
    frame=cv2.resize(frame, (640,480))
    #ウィンドウに画像を出力
    cv2.imshow('OpenCV Camera', frame)
    #ESCかEnterキーでループを抜ける
    k=cv2.waitKey(1)
    
    if k==13 or k==27:
        break

#カメラを開放
cap.release()
#ウィンドウを閉じる
cv2.destroyAllWindows()

