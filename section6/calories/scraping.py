"""
・画像を収集する
・画像をリサイズする

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from icrawler.builtin import BingImageCrawler
import os, glob, re
from PIL import Image
import cv2

#画像を収集する関数
#引数；パス、検索ワード、枚数、
def scraping(path, keyword, num):
    #bing_crawlerのインスタンス生成
    bing_crawler=BingImageCrawler(
        downloader_threads=4,
        storage={'root_dir': path}
    )
    #bing_crawlerのcrawlメソッドでキーワードと枚数を指定して画像を収集
    bing_crawler.crawl(keyword=keyword, max_num=num)
    print(f'{keyword} completed!')


sarad_path="./sarad/*.jpg"
sushi_path="./sushi/*.jpg"
tofu_path="./tofu/*.jpg"

# scraping(path="./image/sarad/", keyword="サラダ", num=300)
# scraping(path="./image/sushi/", keyword="マグロ　寿司", num=300)
scraping(path="./image/tofu/", keyword="麻婆豆腐", num=300)

#画像をリサイズする関数
#引数；パス、サイズ
def resize_image(path, size):
    img_paths=glob.glob(path)
    for img_path in img_paths:
        img=Image.open(img_path)
        img_resized=img.resize((size, size))
        img_resized.save(img_path)
    print(f'{path}: resized!')


resize_image(path=sarad_path, size=75)
resize_image(path=sushi_path, size=75)
resize_image(path=tofu_path, size=75)
