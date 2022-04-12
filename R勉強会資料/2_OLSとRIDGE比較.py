# coding:utf-8

# --------------------------
# (c) yuki nagae @ tryeting
# MIT
# --------------------------

# 最小２乗回帰とリッジ回帰モデルを構築しモデル性能とその中身を比較してみましょう。
# データはボストン・ハウジングデータを使います。

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

# モデルの構築は以下の通りです。
# ツリー系のアルゴリズム（ランダムフォレストや勾配ブースティングなど）を除き、
# 通常、多くの機械学習モデルは、入力ベクトルのスケールを統一させる必要があります。
# ここではその処理をPipelineで組み込んだサンプルです。

# import libraries
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# make pipelines for modeling
pipe_ols = Pipeline([('scl',  StandardScaler()), ('est', LinearRegression())])
pipe_ridge = Pipeline([('scl', StandardScaler()), ('est', Ridge())])

# build models
pipe_ols.fit(X, y.as_matrix().ravel())
pipe_ridge.fit(X, y.as_matrix().ravel())

# get R2 score
y_true = y.as_matrix().ravel()
y_pred_ols = pipe_ols.predict(X)
y_pred_ridge = pipe_ridge.predict(X)

# print the performance
print('R2 score of the OLS model: %.6f' % r2_score(y_true, y_pred_ols))
print('R2 score of the Ridge model: %.6f' % r2_score(y_true, y_pred_ridge))

# OLSとRidgeのどちらが良い予測モデルかをholdout（交差検証）により検証してみましょう。
# またtrain_test_splitのランダムシードの値、リッジ回帰のalphaの値を変化させた時
# （デフォルトの1.0から10.0などへ）のモデルパフォーマンスや、標準偏回帰係数の総和の変化を見てみましょう。
# このデータでは、OLSとリッジ回帰に大きな性能差は見られないと思います。
# ただリッジ回帰のalphaを大きくすると、係数総和が減少していく様子が確認できるはずです。

# import libraries
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 交差検証のためデータを訓練とテストに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10)

# make pipelines for modeling
pipe_ols = Pipeline([('scl',  StandardScaler()), ('est', LinearRegression())])
pipe_ridge = Pipeline([('scl', StandardScaler()), ('est', Ridge(alpha=10.0))])

# build models
pipe_ols.fit(X_train, y_train.as_matrix().ravel())
pipe_ridge.fit(X_train, y_train.as_matrix().ravel())

# 性能指標の表示
print('-----------------------------------------------------')
print('Test Score of OLS : %.6f' % r2_score(y_test, pipe_ols.predict(X_test)))
print('Test Score of Ridge : %.6f' % r2_score(y_test, pipe_ridge.predict(X_test)))

# 回帰係数の総和比較
# リッジ回帰の正則化項の役割把握のため（モデルの性能評価ではありません）
print('-----------------------------------------------------')
print('Absolute Sum of coefficient of OLS  model: %.6f, %.6f' % np.absolute(pipe_ols.named_steps['est'].coef_).sum())
print('Absolute Sum of coefficient of Ridge  model: %.6f' % np.absolute(pipe_ridge.named_steps['est'].coef_).sum())

