#画面に動きがあった部分の検出
import cv2

capture=cv2.VideoCapture(1)
#前回の画像を記録する変数
img_last=None
green=(0,255,0)

while True:
    #画像の取得
    ret, frame=capture.read()
    frame=cv2.resize(frame, (800,600))

    #白黒画像に変換
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray, (9,9), 0)
    img_b=cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]

    #差分の確認
    #<変化なしの場合>
    if img_last is None:
        img_last=img_b
        continue

    #画像ごとの差分を調べる
    frame_diff=cv2.absdiff(img_last, img_b)
    #変化があった部分の輪郭抽出
    contours=cv2.findContours(frame_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    #差分があった部分を画面に描く
    for pt in contours:
        x,y,w,h=cv2.boundingRect(pt)
        #小さな変更点は無視
        if w<30:
            continue
        cv2.rectangle(frame, (x,y), (x+w,y+h), green, 2)

    #今回の画像を保存する
    img_last=img_b
    #画面に表示
    cv2.imshow("Diff Camera", frame)
    cv2.imshow("diff data", frame_diff)
    if cv2.waitKey(1)==13:
        break

capture.release()
cv2.destroyAllWindows()
