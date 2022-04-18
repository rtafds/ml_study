import os
import shutil
import numpy as np
import pandas as pd
import time
import pickle
import gc
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import argparse
from pmlb import fetch_data
from sklearn import datasets




parser = argparse.ArgumentParser()
parser.add_argument("use_dir", nargs="?", default=None)
parser.add_argument("--test", action="store_true")  # デフォルトFalse, --testでTrue
parser.add_argument("--nokfold", action="store_true")  # kfoldをしない。1回のみ。
parser.add_argument("--exhale", action="store_true")  # new_dataを吐き出す
# evaluate only
parser.add_argument("--new_csv",action="store_true")  # デフォルトでは作ったdfは出さない。
parser.add_argument("--score", action="store_true")  # デフォルトはFalse。scoreは出さない。
parser.add_argument("--output_name", default=None)
args = parser.parse_args()

use_dir = args.use_dir
test = args.test
nokfold = args.nokfold
exhale = args.exhale
new_csv = args.new_csv
is_score = args.score
output_name = args.output_name


# fetch data のリスト
reviewed_datasets = [
     'molecular_biology_promoters',
     'car',
     'connect_4',
     'dna',
     '542_pollution',
     '560_bodyfat',
     '1089_USCrime',
     '529_pollen',
     'chess',
     'penguins',
     'bupa',
     'movement_libras',
     'adult',
     'waveform_21',
     'waveform_40',
     'saheart',
     'wine_quality_white',
     'wine_quality_red',
     'irish',
     'mushroom'
 ]

n_splits = 5  # kfoldでcross validationする回数

if test:
    data_kind_list = ["B","M","R"]
    use_data_name = [reviewed_datasets[0], reviewed_datasets[1], reviewed_datasets[4]]
    dfsets = list(map(fetch_data, use_data_name))
else:
    sk_datasets_name = ["load_boston", "load_digits", "load_wine", "fetch_california_housing"]
    sk_datasets = []
    for name in sk_datasets_name:
        print(name)
        data = eval("datasets.{}()".format(name))
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.DataFrame(data.target,columns=["target"])
        df = pd.concat([X,y], axis=1)
        sk_datasets.append(df)
    use_data_name = reviewed_datasets
    dfsets = list(map(fetch_data, use_data_name)) + sk_datasets
    use_data_name = use_data_name + sk_datasets_name
    # "B": binary classifier, "C": multi classifier, "R":regression
    data_kind_list = ["B", "M", "B", "B", "R","R","R","R", "B", "B", "B","M","B","M","M","B","M","M","B","B","R","R","R","R"]

len_dfsets = len(dfsets)

# ディレクトリがなければ作る
this_file_dir = os.path.dirname(os.path.abspath(__file__))
if use_dir==None:
    use_dir = "cross_val"
use_path = this_file_dir + "/" + str(use_dir)
os.makedirs(use_path, exist_ok=True)
train_dir = "/trains/"
test_dir = "/tests/"
train_path = use_path + train_dir
test_path = use_path + test_dir


# 訓練用データ分け
kf = KFold(n_splits=n_splits, shuffle=True)

# ディレクトリの作成
if (os.path.isdir(train_path) == True):
    shutil.rmtree(train_path)
os.makedirs(train_path, exist_ok=True)

if (os.path.isdir(test_path) == True):
    shutil.rmtree(test_path)
os.makedirs(test_path, exist_ok=True)


verification_list = []
for p in range(len_dfsets):
    df = dfsets[p]
    data_kind = data_kind_list[p]
    score_lr_raw_list = []
    score_gb_raw_list = []
    for q, (train_row, test_row) in enumerate(kf.split(df)):
        if nokfold and q==1:  # nokfold=Trueのときは、1回で終わり
            break
                
        train, test = df.iloc[train_row, :], df.iloc[test_row, :]
        # trainとtestを吐き出し
        if exhale:
            train_q_path = train_path + use_data_name[p]
            os.makedirs(train_q_path, exist_ok=True)
            train.to_csv("{}/split{}.csv".format(train_q_path,q), index=False)
            test_q_path = test_path + use_data_name[p]
            os.makedirs(test_q_path, exist_ok=True)
            test.to_csv("{}/split{}.csv".format(test_q_path, q), index=False)   
        
        row_num = train.shape[0]
        col_num = test.shape[1]
        
        if data_kind=="M" or data_kind=="B":
            pipe_lr_raw = Pipeline([("scl",StandardScaler()),("est",LogisticRegression()) ])
            pipe_gb_raw = Pipeline([("scl",StandardScaler()),("est",GradientBoostingClassifier()) ])
        elif data_kind=="R":
            pipe_lr_raw = Pipeline([("scl",StandardScaler()),("est",LinearRegression()) ])
            pipe_gb_raw = Pipeline([("scl",StandardScaler()),("est",GradientBoostingRegressor()) ])
        X_train_raw = train.iloc[:,:-1]
        y_train_raw  = train.iloc[:,-1]
        X_test = test.iloc[:,:-1]
        y_test = test.iloc[:,-1]
        
        # 学習してスコアを出す
        try:
            pipe_lr_raw.fit(X_train_raw, y_train_raw)
            y_pred_lr_raw = pipe_lr_raw.predict(X_test)
            if data_kind=="B" or data_kind=="M":
                accuracy_score_lr_raw = accuracy_score(y_test, y_pred_lr_raw)
                score_lr_raw = accuracy_score_lr_raw      
            elif data_kind=="R":
                r2_score_lr_raw = r2_score(y_test, y_pred_lr_raw)
                score_lr_raw = r2_score_lr_raw
        except Exception as e:
            print(e)
            score_lr_raw=np.nan
        try:
            pipe_gb_raw.fit(X_train_raw, y_train_raw)
            y_pred_gb_raw = pipe_gb_raw.predict(X_test)
            if data_kind=="B" or data_kind=="M":
                accuracy_score_gb_raw = accuracy_score(y_test, y_pred_gb_raw)       
                score_gb_raw = accuracy_score_gb_raw
            elif data_kind=="R":
                r2_score_gb_raw = r2_score(y_test, y_pred_gb_raw)
                score_gb_raw = r2_score_gb_raw
        except Exception as e:
            print(e)
            score_gb_raw=np.nan
        score_lr_raw_list.append(score_lr_raw)
        score_gb_raw_list.append(score_gb_raw)
        
    del pipe_gb_raw
    del pipe_lr_raw
    del train
    del test
    gc.collect()
    
    # 値を変換していく。
    score_lr_raw_mean = np.array(score_lr_raw_list).mean() # n_splitsで平均
    score_gb_raw_mean = np.array(score_gb_raw_list).mean() # 1個の数字


    verification = [ use_data_name[p], row_num, col_num] + \
        [data_kind,score_lr_raw_mean, score_gb_raw_mean]
    verification_list.append(verification)

    colnames = [ "use_data", "row_num", "col_num"] + \
        ["data_kind", "raw_r2/accuracy_Logistic/Linear"] + \
        ["raw_r2/accuracy_GradientBoosting"]
        
    verification_df = pd.DataFrame(verification_list, columns=colnames)

    results_path = use_path + "/results/"

    os.makedirs(results_path, exist_ok=True)
    if output_name==None:
        output_name="evaluation_all.csv"
    else:
        output_name = str(output_name)
    verification_df.to_csv(results_path+output_name, index=False)






