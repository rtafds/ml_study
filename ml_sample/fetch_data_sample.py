import os
import shutil
import numpy as np
import pandas as pd
import time
import pickle
import gc
import warnings
warnings.filterwarnings('ignore')
import argparse
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

from pmlb import fetch_data

# 以下は引数
# python cross_validation_synthetic_data.py ./data --evaluate --output_name aaa.csv
# みたいに指定する
parser = argparse.ArgumentParser()
parser.add_argument("use_dir", nargs="?", default=None)

parser.add_argument("--test", action="store_true")  # デフォルトFalse, --testでTrue
parser.add_argument("--nokfold", action="store_true")  # kfoldをしない。1回のみ。
parser.add_argument("--output_name", default=None)
args = parser.parse_args()

use_dir = args.use_dir
test = args.test
nokfold = args.nokfold
output_name = args.output_name


n_splits = 5  # kfoldでcross validationする回数

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

# pmlbの全データリスト
pmlb_table = pd.read_table(this_file_dir + "/pmlb_datasets_now.tsv")
if test:
    use_data = pmlb_table.iloc[[1,136],:]
    
else:
    # いろいろ条件でいじって
    # もとのcsvみていろいろといじるといい。
    use_data = pmlb_table.query("n_instances<50000")  # データ数が50000以下のもの
use_data_name = list(use_data["dataset"])
data_kind_list = list(use_data["task"])
data_feature_colnames = list(use_data.iloc[:,1:].columns)

len_dfsets = len(use_data_name)

# -------------------- 学習 --------------------


# 訓練用データ分け
kf = KFold(n_splits=n_splits, shuffle=True)

# ディレクトリの作成
if (os.path.isdir(train_path) == True):
    shutil.rmtree(train_path)
os.makedirs(train_path, exist_ok=True)

if (os.path.isdir(test_path) == True):
    shutil.rmtree(test_path)
os.makedirs(test_path, exist_ok=True)
models_dir = "/models_cross/"
models_path = use_path + models_dir
if (os.path.isdir(models_path) == True):
    shutil.rmtree(models_path)
os.makedirs(models_path, exist_ok=True)

all_times = []
for p in range(len_dfsets):
    # datasets毎に実行する
    df = fetch_data(use_data_name[p])
    times_split = []
    for q, (train_row, test_row) in enumerate(kf.split(df)):
        if nokfold and q==1:  # nokfold=Trueのときは、1回で終わり
            break
                
        train, test = df.iloc[train_row, :], df.iloc[test_row, :]
        print(train)
