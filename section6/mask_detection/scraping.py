import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from icrawler.builtin import BingImageCrawler
import os
import re
from PIL import Image
import glob


# 画像を収集するメソッド
# 引数は画像を保存するパスpath、検索ワードkeyword、収集する枚数num

def scraping(path, keyword, num):

    bing_crawler=BingImageCrawler(
    downloader_threads=4,
    storage={'root_dir': path}
    )

    #検索ワードにkeywordを入れたときに得られる画像をnum枚収集
    bing_crawler.crawl(
        keyword=keyword,
        max_num=num
    )
    print(f'{keyword}: scraping completed!')


#ファイルの形式はjpegなので、ファイル名には必ず拡張子.jpgがつく
gather_path='./gather/*.jpg'
mask_people_path='./mask_people/*.jpg'

keywords=['集合写真', 'マスク 東京']
num=300

scraping('./image/gather/', keywords[0], num)
scraping('./image/mask_on_entry/', keywords[1], num)

# """
# 画像をリサイズするメソッド
# 引数は保存したいパスpath=フォルダ名+フォーマット名、変更後のサイズの幅と高さw,h
#
# *リサイズしたい画像はパスで指定される
# """

def resize_image(path, w, h):
    img_paths=glob.glob(path)

    for img_path in img_paths:
        #画像ファイルに変換
        img=Image.open(img_path)
        #指定したサイズでリサイズをする
        img_resized=img.resize((w,h))

        #リサイズした画像を上書き保存、同じパスを指定
        img_resized.save(img_path)
    print(f'{path}: resized!')

#サイズは300x300で指定
width=300
height=300

resize_image(gather_path, width, height)
resize_image(mask_people_path, width, height)
