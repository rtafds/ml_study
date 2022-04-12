# coding:utf-8

# --------------------------
# (c) yuki nagae @ tryeting
# MIT
# --------------------------

# ここではモデル構築の大きな流れを掴みましょう。まずは、サンプルデータ（回帰用）を読み込みます。

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

# アンサンブル学習（ブースティング）の一つである勾配ブースティンレグレッサーを読み込み学習（fit）します。
# 予測値と正解データの整合性をR2値で評価しています。データを読み込み、学習させ、評価する流れを掴んで下さい。

# import libraries
from sklearn.ensemble import GradientBoostingRegressor

# build the model
est = GradientBoostingRegressor(max_depth=3, random_state=42)
est.fit(X, y.as_matrix().ravel())

# check the model performance by R2 socre
from sklearn.metrics import r2_score
y_true = y.as_matrix().ravel()
y_pred = est.predict(X)
r2 = r2_score(y_true, y_pred)
print('R2 score of the descriptive model: %.3f' % r2)

# モデル構築の流れは以上なのですが、予測を目的とした場合、モデル構築のステップは増加します。
# それは未知データへの当てはまりに興味があるためです。予測上の最大の敵は過学習であり、
# この過学習へ対処するためのステップが増えると理解して下さい。交差検証はその対処法の一つです。

# import libraries
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


# 交差検証のためデータを訓練とテストに分割
# 訓練を80%, テストを20%に分割
# 訓練とテストにランダム分割するだけの単純な交差検証はhold-outと呼ばれる
X_train, X_test, y_train, y_test = train_test_split(X, y)

# 比較用に二つのパラメータ違いのモデルを構築
# standard tree model
est1 = GradientBoostingRegressor(max_depth=3, random_state=42)
est1.fit(X_train, y_train.as_matrix().ravel())
# deeper tree model
est2 = GradientBoostingRegressor(max_depth=10, random_state=42) 
est2.fit(X_train, y_train.as_matrix().ravel())

# モデルパフォーマンス指標(R2とする)を取得
from sklearn.metrics import r2_score
# for training data
r2_est1_train = r2_score(y_train.as_matrix().ravel(), est1.predict(X_train))
r2_est2_train = r2_score(y_train.as_matrix().ravel(), est2.predict(X_train))
# for test data
r2_est1_test = r2_score(y_test.as_matrix().ravel(), est1.predict(X_test))
r2_est2_test = r2_score(y_test.as_matrix().ravel(), est2.predict(X_test))


# 性能指標の表示
# 以下のスコアをどのように評価すべきか？ --> Keyword: overfitting, train test gap
print('-----------------------------------------------------')
print('Train Score(est1, est2) : (%.3f, %.3f)' % (r2_est1_train, r2_est2_train))
print('Test Score(est1, est2) : (%.3f, %.3f)' % (r2_est1_test, r2_est2_test))

# est1のパラメータ条件で最終モデルを構築
est1.fit(X, y.as_matrix().ravel())
print('-----------------------------------------------------')
print(est1)

# 過学習モデルも参考のため構築
est2.fit(X,y.as_matrix().ravel())
print('-----------------------------------------------------')
print(est2)