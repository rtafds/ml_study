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
from sdv.tabular import GaussianCopula
from sdv.tabular import CTGAN
from sdv.tabular import CopulaGAN
from sdv.tabular import TVAE
from sklearn import datasets


# 完全に合成データの時に使っているコード。学習パートは完全に合成データを学習する部分。evaluateが参考になるかも。

# 以下は引数
# python cross_validation_synthetic_data.py ./data --evaluate --output_name aaa.csv
# みたいに指定する
parser = argparse.ArgumentParser()
parser.add_argument("use_dir", nargs="?", default=None)
parser.add_argument("--learn", action="store_true")  # 学習だけモード
parser.add_argument("--evaluate", action="store_true")  # evaluateだけモード
parser.add_argument("--test", action="store_true")  # デフォルトFalse, --testでTrue
parser.add_argument("--nokfold", action="store_true")  # kfoldをしない。1回のみ。
# evaluate only
parser.add_argument("--exhale",action="store_true")  # 合成データを出力
parser.add_argument("--score", action="store_true")  # デフォルトはFalse。scoreは出さない。
parser.add_argument("--new_amount", default=None)  # 作って学習させる量 デフォルトは元データと同じ量
parser.add_argument("--output_name", default=None)
args = parser.parse_args()

use_dir = args.use_dir
is_learn = args.learn
is_evaluate = args.evaluate
test = args.test
nokfold = args.nokfold
exhale = args.exhale
is_score = args.score
new_amount = args.new_amount
output_name = args.output_name

# デフォルトは両方やる
if is_learn==False and is_evaluate==False:
    is_learn = True
    is_evaluate = True



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
pmlb_table = pd.read_table(this_file_dir + "/pmlb_datasets.tsv")
if test:
    use_data = pmlb_table.iloc[[1,136],:]
    
else:
    # いろいろ条件でいじって
    use_data = pmlb_table.query("n_instances<50000")  # データ数が50000以下のもの
use_data_name = list(use_data["dataset"])
data_kind_list = list(use_data["task"])
data_feature_colnames = list(use_data.iloc[:,1:].columns)

len_dfsets = len(use_data_name)

# -------------------- 学習 --------------------
if is_learn:  

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
            # trainとtestを吐き出し

            train_q_path = train_path + use_data_name[p]
            os.makedirs(train_q_path, exist_ok=True)
            train.to_csv("{}/split{}.csv".format(train_q_path,q), index=False)
            test_q_path = test_path + use_data_name[p]
            os.makedirs(test_q_path, exist_ok=True)
            test.to_csv("{}/split{}.csv".format(test_q_path, q), index=False)          
                

            models = [GaussianCopula(), CTGAN(), CopulaGAN(), TVAE()]

            times = []
            len_models = len(models)
            models_q_path = models_path + use_data_name[p] + "/split" + str(q)
            os.makedirs(models_q_path, exist_ok=True)
            for i in range(len_models):
                start = time.time()
                models[i].fit(train)
                elapsed_time = time.time() - start
                times.append(elapsed_time)
                # モデルの書き出し
                with open("{}/model{}.pickle".format(models_q_path, i), "wb") as f:
                    pickle.dump(models[i], f)
                print("fitting complete data:{} split:{} model:{}".format(use_data_name[p], q, i))
            times_split.append(times)
            del models
            del train
            del test
        print(times_split)
        times_mean = np.array(times_split).mean(axis=0)
        all_times.append(times_mean)
        del df
        gc.collect()
        # 計算時間の表の書き出し。毎回書き出す。
        times_df = pd.DataFrame(all_times)
        times_df.to_csv(models_path+"times.csv", index=False)


# -------------------- 学習終わり ------------------------


# -------------------- 評価開始 -----------------------
if is_evaluate:
    models_name = ["GaussianCopula","CTGAN","CopulaGAN","TVAE"]
    #models_name = ["com_dims=(256,256)","com_dims=(512,512)","com_dims=(1024,1024)",\
    #    "com_dims=(2048,2048)","emb_dim=256","emb_dim=512","emb_dim=1024","emb_dim=2048"]

    if exhale:
        synthetic_dir = "/synthetic/"
        synthetic_path = use_path + synthetic_dir
        if (os.path.isdir(synthetic_path) == True):
            shutil.rmtree(synthetic_path)
        os.makedirs(synthetic_path, exist_ok=True)
        
    models_dir = "/models_cross/"
    models_path = use_dir + models_dir
    #times_df = pd.read_csv(models_path+"times.csv")

    verification_list = []
    for p in range(len_dfsets):
        data_kind = data_kind_list[p]
        score_lr_raw_list = []
        score_gb_raw_list = []
        score_lr_new_list = []
        score_gb_new_list = []
        all_sizes = []
        for q in range(n_splits):
            if nokfold and q==1:  # nokfold=Trueのときは、1回で終わり
                break
            
            # trainとtestを読み込み
            train_q_path = train_path + use_data_name[p]
            train = pd.read_csv("{}/split{}.csv".format(train_q_path,q) )
            test_q_path = test_path + use_data_name[p]
            test = pd.read_csv("{}/split{}.csv".format(test_q_path, q) )   
            
            raw_df = train
            row_num = raw_df.shape[0]
            col_num = raw_df.shape[1]
            
            if data_kind=="classification":
                pipe_lr_raw = Pipeline([("scl",StandardScaler()),("est",LogisticRegression()) ])
                pipe_gb_raw = Pipeline([("scl",StandardScaler()),("est",GradientBoostingClassifier()) ])
            elif data_kind=="regression":
                pipe_lr_raw = Pipeline([("scl",StandardScaler()),("est",RandomForestRegressor()) ])
                pipe_gb_raw = Pipeline([("scl",StandardScaler()),("est",GradientBoostingRegressor()) ])
            X_train_raw = raw_df.iloc[:,:-1]
            y_train_raw  = raw_df.iloc[:,-1]
            X_test = test.iloc[:,:-1]
            y_test = test.iloc[:,-1]
            
            # 学習してスコアを出す
            try:
                pipe_lr_raw.fit(X_train_raw, y_train_raw)
                y_pred_lr_raw = pipe_lr_raw.predict(X_test)
                if data_kind=="classification":
                    accuracy_score_lr_raw = accuracy_score(y_test, y_pred_lr_raw)
                    score_lr_raw = accuracy_score_lr_raw        
                elif data_kind=="regression":
                    r2_score_lr_raw = r2_score(y_test, y_pred_lr_raw)
                    score_lr_raw = r2_score_lr_raw
            except Exception as e:
                print(e)
                score_lr_raw=np.nan
            try:
                pipe_gb_raw.fit(X_train_raw, y_train_raw)
                y_pred_gb_raw = pipe_gb_raw.predict(X_test)
                if data_kind=="classification":
                    accuracy_score_gb_raw = accuracy_score(y_test, y_pred_gb_raw)       
                    score_gb_raw = accuracy_score_gb_raw
                elif data_kind=="regression":
                    r2_score_gb_raw = r2_score(y_test, y_pred_gb_raw)
                    score_gb_raw = r2_score_gb_raw
            except Exception as e:
                print(e)
                score_gb_raw=np.nan
            score_lr_raw_list.append(score_lr_raw)
            score_gb_raw_list.append(score_gb_raw)
            
            # 合成データで検証
            # モデル読み込み
            len_models = len(models_name)
            models_q_path = models_path + use_data_name[p] + "/split" + str(q)
            if exhale:
                synthetic_q_path = synthetic_path + use_data_name[p]
                os.makedirs(synthetic_q_path, exist_ok=True)
                
            # モデルループ
            score_lr_new_list_i = []
            score_gb_new_list_i = []
            sizes = []
            if is_score:
                scores_list_list = []
                scores_mean_list = []
            for i in range(len_models):
                with open("{}/model{}.pickle".format(models_q_path, i), "rb") as f:
                    model = pickle.load(f)
                    size = os.path.getsize(f.name)       
                    sizes.append(size)
                if new_amount==None:
                    synthetic_df = model.sample(row_num)
                    evaluate_df = synthetic_df   
                else:
                    # 引数が入った場合
                    new_amount = int(new_amount)
                    if row_num <= new_amount:
                        synthetic_df = model.sample(new_amount)
                        evaluate_df = synthetic_df.iloc[row_num, :]
                    else:
                        synthetic_df = model.sample(row_num)
                        evaluate_df = synthetic_df
                        
                    
                # 新規データ吐き出し
                if exhale:
                    synthetic_i_path = synthetic_q_path + "/split" + str(q)
                    os.makedirs(synthetic_i_path, exist_ok=True)
                    synthetic_df.to_csv("{}/{}.csv".format(synthetic_i_path, models_name[i]), index=False)
                
                if is_score:
                    scores = evaluate(evaluate_df, raw_df, aggregate=False)
                    scores_mean = scores["normalized_score"].mean()
                    scores_list = list(scores["normalized_score"])
                    scores_list_list.append(scores_list)
                    scores_mean_list.append(scores_mean)
                
                # 機械学習
                if data_kind=="classification":
                    pipe_lr_new = Pipeline([("scl",StandardScaler()),("est",LogisticRegression()) ])
                    pipe_gb_new = Pipeline([("scl",StandardScaler()),("est",GradientBoostingClassifier()) ])
                    
                elif data_kind=="regression":
                    pipe_lr_new = Pipeline([("scl",StandardScaler()),("est",RandomForestRegressor()) ])
                    pipe_gb_new = Pipeline([("scl",StandardScaler()),("est",GradientBoostingRegressor()) ])

                X_train_new = synthetic_df.iloc[:,:-1]
                y_train_new = synthetic_df.iloc[:, -1]

                try:
                    pipe_lr_new.fit(X_train_new, y_train_new)
                    y_pred_lr_new = pipe_lr_new.predict(X_test)
                    y_pred_lr_new = pipe_lr_new.predict(X_test)
                    if data_kind=="classification":
                        accuracy_score_lr_new = accuracy_score(y_test, y_pred_lr_new)
                        score_lr_new = accuracy_score_lr_new
                    elif data_kind=="regression":
                        r2_score_lr_new = r2_score(y_test, y_pred_lr_new)
                        score_lr_new = r2_score_lr_new
                except Exception as e:
                    print(e)
                    score_lr_new = 0

                try:
                    pipe_gb_new.fit(X_train_new, y_train_new)
                    y_pred_gb_new = pipe_gb_new.predict(X_test)
                    if data_kind=="classification":
                        accuracy_score_gb_new = accuracy_score(y_test, y_pred_gb_new)
                        score_gb_new = accuracy_score_gb_new
                        
                    elif data_kind=="regression":
                        r2_score_gb_new = r2_score(y_test, y_pred_gb_new)
                        score_gb_new = r2_score_gb_new
                except Exception as e:
                    print(e)
                    score_gb_new = 0
                
                score_lr_new_list_i.append(score_lr_new)
                score_gb_new_list_i.append(score_gb_new)
                print("evaluate complete data:{} split:{} model:{}".format(use_data_name[p], q, i))
            score_lr_new_list.append(score_lr_new_list_i)
            score_gb_new_list.append(score_gb_new_list_i)
            all_sizes.append(sizes)
            
        del pipe_gb_new
        del pipe_gb_raw
        del pipe_lr_new
        del pipe_lr_raw
        del synthetic_df
        del train
        del test
        gc.collect()
        
        # 値を変換していく。
        score_lr_raw_mean = np.array(score_lr_raw_list).mean() # n_splitsで平均
        score_gb_raw_mean = np.array(score_gb_raw_list).mean() # 1個の数字
        score_lr_new_mean_array = np.array(score_lr_new_list).mean(axis=0) # 各モデルでn_splits平均
        score_gb_new_mean_array = np.array(score_gb_new_list).mean(axis=0) # len_modelのarray
        sizes_array = np.array(all_sizes).mean(axis=0)
        
        if is_score:
            many_scores_mean = list(np.array(scores_list_list).mean(axis=0))

        for i in range(len_models):
            verification = [ models_name[i], use_data_name[p], row_num, col_num, 1 , sizes_array[i]] + \
                [score_lr_raw_mean, score_lr_new_mean_array[i], score_gb_raw_mean, score_gb_new_mean_array[i]]+\
                list(use_data.iloc[p,1:])
            #verification = [ models_name[i], use_data_name[p], row_num, col_num, times_df.iloc[p,i] , sizes_array[i]] + \
            #    [score_lr_raw_mean, score_lr_new_mean_array[i], score_gb_raw_mean, score_gb_new_mean_array[i]]+\
            #    list(use_data.iloc[p,1:])
            if is_score:
                verification = verification + [scores_mean_list[i]] + many_scores_mean[i]
            verification_list.append(verification)

        colnames = ["model", "use_data", "row_num", "col_num", "fit_time","size"] + \
            ["raw_r2/accuracy_Logistic/Linear", "new_r2/accuracy_Logistic/Linear"] + \
            ["raw_r2/accuracy_GradientBoosting", "new_r2/accuracy_GradientBoosting"] + \
            data_feature_colnames
            
        if is_score:
            scores_name = list(scores["metric"])
            colnames = colnames + ["scores_mean"] + scores_name

        verification_df = pd.DataFrame(verification_list, columns=colnames)

        results_path = use_path + "/results/"

        os.makedirs(results_path, exist_ok=True)
        if output_name==None:
            output_name="evaluation_all.csv"
        else:
            output_name = str(output_name)
        verification_df.to_csv(results_path+output_name, index=False)






