"""
画像ファイルを読んでNumpy形式に変換して
画像にラベルをつけたデータとして保存する
"""

import numpy as np
from PIL import Image
import random, os, glob

#保存するファイル名
outfile="./image/photos.npz"

#利用する写真の枚数, 画像サイズ
max_photo=100
photo_size=32

#画像データ
x=[]
#ラベルデータ
y=[]

def main():
    #各画像のフォルダを読む
    glob_files("./image/sushi", 0)
    glob_files("./image/sarad", 1)
    glob_files("./image/tofu", 2)
    #ファイルへ保存
    np.savez(outfile, x=x, y=y)
    print("saved: "+outfile, len(x))

#path以下の画像を読み込む
def glob_files(path, label):
    files=glob.glob(path+"/*.jpg")
    random.shuffle(files)

    #ファイルを処理
    num=0
    for f in files:
        if num>=max_photo:
            break
        num+=1
        #画像ファイルを読む
        img=Image.open(f)
        img=img.convert("RGB")
        img=img.resize((photo_size, photo_size))
        img=np.asarray(img)
        x.append(img)
        y.append(label)

if __name__=='__main__':
    main()
