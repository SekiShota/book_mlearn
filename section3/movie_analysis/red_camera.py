#赤色成分だけ表示
import cv2
import numpy as np

#カメラからの入力開始
capture=cv2.VideoCapture(1)
while True:
    #画像を取得
    ret, frame=capture.read()

    #画像を縮小表示
    frame=cv2.resize(frame, (500,300))

    #BGRのB,Gを0にする
    # frame[:,:,0]=0
    # frame[:,:,1]=0
    # frame[:,:,2]=0

    #ウィンドウに画像を出力
    cv2.imshow("Red Camera", frame)

    #Enterキーでループ抜ける
    if cv2.waitKey(1)==13:
        break

#カメラの開放
capture.release()
#ウィンドウを破棄
cv2.destroyAllWindows()
