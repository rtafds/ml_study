# coding:utf-8

# --------------------------
# (c) yuki nagae @ tryeting
# MIT
# --------------------------

# OLS、リッジ回帰、ランダムフォレスト、勾配ブースティングのアルゴリズム性能を比較してみましょう。
# データはボストン・ハウジングデータを使いましょう。

# import the data for regression
import pandas as pd
from sklearn.datasets import load_boston
dataset = load_boston()

# set dataframe
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.DataFrame(dataset.target, columns=['y'])

# check the shape
print('----------------------------------------------------------------------------------------')
print('X shape: (%i,%i)' %X.shape)
print('y shape: (%i,%i)' %y.shape)
print('----------------------------------------------------------------------------------------')
print(y.describe())
print('----------------------------------------------------------------------------------------')
print(X.join(y).head())

# ツリー系のアルゴリズム（ランダムフォレストや勾配ブースティングなど）を除き、
# 通常、多くの機械学習モデルは、入力ベクトルのスケールを統一させる必要があります。
# ここではその処理をPipelineで組み込んでいます。

# import libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# 交差検証(holdout)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# make pipelines for modeling
pipe_ols = Pipeline([('scl',StandardScaler()),('est',LinearRegression())])
pipe_ridge = Pipeline([('scl',StandardScaler()),('est',Ridge())])
pipe_rf = Pipeline([('scl',StandardScaler()),('est',RandomForestRegressor(random_state=42))])
pipe_gbr = Pipeline([('scl',StandardScaler()),('est',GradientBoostingRegressor(random_state=42))])

# build models
pipe_ols.fit(X_train, y_train.as_matrix().ravel())
pipe_ridge.fit(X_train, y_train.as_matrix().ravel())
pipe_rf.fit(X_train, y_train.as_matrix().ravel())
pipe_gbr.fit(X_train, y_train.as_matrix().ravel())

# get R2 score
y_true = y_test.as_matrix().ravel()

# print the performance
print('R2 score of the OLS: %.6f' % r2_score(y_true, pipe_ols.predict(X_test)))
print('R2 score of the Ridge: %.6f' % r2_score(y_true, pipe_ridge.predict(X_test)))
print('R2 score of the RandomForest: %.6f' % r2_score(y_true, pipe_rf.predict(X_test)))
print('R2 score of the GradinetBoostingRegressor: %.6f' % r2_score(y_true, pipe_gbr.predict(X_test)))

# 最後にOLSとランダムフォレストの予実プロットを作成してみましょう。大きな誤差が出にくくなっているのが確認できます。

# MUST install plotly before this execution: conda install plotly
import plotly.offline as py
from plotly.graph_objs import *
plotly.offline.init_notebook_mode(connected=False)

trace1 = Scatter(x=pipe_ols.predict(X_test), y=y_true, mode='markers', name='OLS')
trace2 = Scatter(x=pipe_gbr.predict(X_test), y=y_true, mode='markers', name='GradientBoosting')

data = [trace1, trace2]
py.offline.plot(data)