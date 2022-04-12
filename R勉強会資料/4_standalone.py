# coding:utf-8
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.externals import joblib

# 第一引数を学習用データのパスに

# 引数の代入
args = sys.argv

# データ読み込み〜説明変数(X)、目的変数(y)に分割
df_test  = pd.read_csv(args[1], header=0)  # headerなしの場合はNoneを指定
X = df_test.ix[:, 1:]
y = df_test.ix[:, 0]

# Holdout検証用データ起こし
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20, random_state=1)

# set pipelines for different algorithms
pipe = Pipeline([('scl',StandardScaler()),('est',GradientBoostingRegressor(random_state=1))])

# 学習(gradient boosting)
pipe.fit(X_train, y_train.as_matrix().ravel())

df_pred = pd.DataFrame({
    'test': y_test.as_matrix().ravel(),
    'predicted': pipe.predict(X_test)})

df_pred.to_csv("./RESULT.csv", index=False)
X_test.to_csv("./TESTDATA.csv", index=False)

# 評価〜出力
R2 = pd.DataFrame(
    [r2_score(y_test.as_matrix().ravel(), pipe.predict(X_test))],
    columns=['R2_score'])

R2.to_csv('./R2SCORE.csv', index=False)

# モデルを保存する
joblib.dump(pipe,'./MODEL.sav')

print('学習が終了しました')
