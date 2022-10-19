"""
データベースに100件分のデータを挿入する
"""

import sqlite3
import random
N=5000

dbpath="./hw.sqlite3"

def insert_db(conn):
    #身長と体重のデータを生成
    height=random.randint(130, 180)
    weight=random.randint(30, 100)
    #体型データはBMIに基づいて自動生成
    typeNo=1
    bmi=weight/(height/100)**2
    if bmi<18.5:
        typeNo=0
    elif bmi<25:
        typeNo=1
    elif bmi<30:
        typeNo=2
    elif bmi<35:
        typeNo=3
    elif bmi<40:
        typeNo=4
    else:
        typeNo=5
    #SQLと値を指定してDBに値を挿入
    sql='''
        INSERT INTO person(height, weight, typeNo)
        VALUES(?,?,?)
    '''
    values=(height, weight, typeNo)
    print(values)
    conn.executemany(sql, [values])

#DBに接続して100件のデータを挿入
with sqlite3.connect(dbpath) as conn:
    #データを100件挿入
    for i in range(N):
        insert_db(conn)

    #トータルで挿入した行数を調べる, fetchoneは１行取得する
    c=conn.execute('SELECT count(*) FROM person')
    cnt=c.fetchone()
    print(cnt[0])
