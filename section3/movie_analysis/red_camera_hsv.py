import cv2
import numpy as np

#カメラからの入力を開始
capture=cv2.VideoCapture(1)

while True:
    ret, frame=capture.read()
    #画像の縮小
    frame=cv2.resize(frame, (800, 600))

    #色空間をHSVに変換、色相・彩度・明度で表現する
    hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)
    #HSVを分割
    h=hsv[:,:,0]
    s=hsv[:,:,1]
    v=hsv[:,:,2]

    #赤色っぽい画素のみ抽出して白くしている,255
    img=np.zeros(h.shape, dtype=np.uint8)
    img[((h<50)|(h>200))&(s>100)]=255

    #ウィンドウに画像を出力
    cv2.imshow("RED Camera", img)
    if cv2.waitKey(1) == 13:
        break

capture.release()
cv2.destroyAllWindows()
