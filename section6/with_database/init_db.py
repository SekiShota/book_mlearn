"""
データベース作成>>>テーブル作成
id: 顧客id
height: 身長
weight: 体重
typeNo: 体型のタイプ
"""

import sqlite3

#dpathで指定した名前でデータベースが作成される
dpath='./hw.sqlite3'
sql='''
    CREATE TABLE IF NOT EXISTS person(
        id INTEGER PRIMARY KEY,
        height NUMBER,
        weight NUMBER,
        typeNo INTEGER
    )

'''

#データベースにアクセスして、sql文実行(テーブル作成)
with sqlite3.connect(dpath) as conn:
    conn.execute(sql)
