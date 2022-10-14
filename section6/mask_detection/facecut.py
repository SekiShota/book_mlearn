import cv2, dlib, sys, glob, pprint

#入力ディレクトリの指定
#gather: マスクなしの人が写った集合写真
#mask_people: マスクをつけた人が写った画像
# indir="./image/gather"
indir="./image/mask_people"

#出力ディレクトリの指定
#mask_off: マスクなし
#mask_on: マスクあり
# outdir="./image/mask_off"
outdir="./image/mask_on"

#暫定的な画像のID
fid=1000
#入力画像をリサイズするかどうか
flag_resize=False


#Dlibを始める（Dlibは画像から顔の検出をするためのツール）
detector=dlib.get_frontal_face_detector()

#顔画像を取得して保存
def get_face(fname):
    global fid
    img=cv2.imread(fname)
    #画像のサイズが大きい時はリサイズする
    if flag_resize:
        img=cv2.resize(img, None, fx=0.2, fy=0.2)

    #顔検出, 検出した部分の矩形の座標を取得している
    dets=detector(img, 1)
    for k,d in enumerate(dets):
        pprint.pprint(d)
        x1=int(d.left())
        y1=int(d.top())
        x2=int(d.right())
        y2=int(d.bottom())
        im=img[y1:y2, x1:x2]

        #50 x 50にリサイズ
        try:
            im=cv2.resize(im, (50,50))
            # print('resized!')
        except:
            # print('Not resized!')
            continue

        #保存する
        out=outdir+"/"+str(fid)+".jpg"
        cv2.imwrite(out, im)
        fid+=1

#ファイルを列挙して繰り返して顔検出を行う
files=glob.glob(indir+"/*")
for f in files:
    print(f)
    get_face(f)
print("ok!")
