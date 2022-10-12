import cv2

# VideoCapture オブジェクトを取得します
# デフォルトでは0が内蔵カメラのIDだが、Snap Cameraを使用している影響でIDは1となっていた
# editorのterminalではなく、単体のterminalから実行するとできる
capture = cv2.VideoCapture(1)

# capture関数からカメラからの読み込みができているか判定
# isOpend関数の戻り値はTrue, False
# print(capture.isOpened())

while(True):
    #read関数の引数はretとframe
    #retは読み込みができているかの判定でTrue, False
    #frameはカメラが捉えた情報
    ret, frame = capture.read()

    #ウィンドウに表示
    cv2.imshow('Mask Checker',frame)

    #qをキーボードで入力するとカメラ停止
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#メモリを解放する
capture.release()

#ウィンドウを閉じる
cv2.destroyAllWindows()
