import os
import shutil
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import argparse

models_name = ["GaussianCopula","CTGAN","CopulaGAN","TVAE"]
dpi = 300

this_file_dir = os.path.dirname(os.path.abspath(__file__))
#this_file_dir = "/home/mamitsu/Github/synthetic_verification"

# 引数の処理
parser = argparse.ArgumentParser()
parser.add_argument("input_name", nargs="?", default=None)
parser.add_argument("--output_name", default=None)
parser.add_argument("--score",action="store_true")
args = parser.parse_args()

input_name = args.input_name
output_name = args.output_name
score = args.score

results_dir = "/results/"
results_path = this_file_dir + results_dir
if input_name==None:
    input_name = "evaluation_machine_all.csv"
else:
    input_name = str(input_name)
results = pd.read_csv(results_path+input_name)


results["amount_of_data"] = results["row_num"] * results["col_num"]
results["diff_Logistic/Linear"] = results["new_r2/accuracy_Logistic/Linear"] -  results["raw_r2/accuracy_Logistic/Linear"]
results["diff_GradientBoosting"] = results["new_r2/accuracy_GradientBoosting"] -  results["raw_r2/accuracy_GradientBoosting"]
results_c = results.query("data_kind=='B' or data_kind=='M'")
results_r = results.query("data_kind=='R'")


# モデルごとの結果
results_model = []
results_model_c = []
results_model_r = []

len_models = len(models_name)
for i in range(len_models):
    results_ = results.query("model==@models_name[@i]")
    results_model.append(results_)
    results_c_ = results_c.query("model==@models_name[@i]")
    results_model_c.append(results_c_)
    results_r_ = results_r.query("model==@models_name[@i]")
    results_model_r.append(results_r_)
    
    
if output_name==None:
    output_name = "fig/"
fig_path = results_path + "/" + output_name + "/"

os.makedirs(fig_path, exist_ok=True)


# 計算時間の表

marker_list = ["o","o","o","o"]
c_list = ["#0000FF","#FF0000","#FF00FF","#00FF00"]
plt_num = (2,2)  # subplots_num
plt.rcParams["figure.figsize"]=[20,20]
plt.subplots_adjust(wspace=0.4, hspace=0.6)
fig, ax = plt.subplots(plt_num[0],plt_num[1])

# figごとにまとめる
use_col_fig = [["col_num","row_num"],["amount_of_data", "use_data"]]
name_fig = [["number of columns", "number of rows"],["number of data","data name"]]
for p in range(plt_num[0]):
    for q in range(plt_num[1]):
        
        
        
        for i in range(len_models):
            ax[p,q].scatter(results_model[i][use_col_fig[p][q]], results_model[i]["fit_time"], label=models_name[i], \
                    marker=marker_list[i], c=c_list[i])

        ax[p,q].set_title("{} - fit time ".format(name_fig[p][q]))
        ax[p,q].set_xlabel("{}".format(name_fig[p][q]))
        ax[p,q].set_ylabel("fitting time(s)")
        if p==0 and q==0:
            ax[p,q].legend(bbox_to_anchor=(-0.3, 1), loc="upper left",borderaxespad=0)
        if p==1 and q==1:
            ax[p,q].tick_params(axis='x', labelrotation=90)

#plt.show()
plt.savefig(fig_path+"fit_time.png", format="png",dpi=dpi)

# データサイズの表

marker_list = ["o","o","o","o"]
c_list = ["#0000FF","#FF0000","#FF00FF","#00FF00"]
plt_num = (2,2)  # subplots_num
#plt.rcParams["figure.figsize"]=[30,10]

fig, ax = plt.subplots(plt_num[0],plt_num[1])

# figごとにまとめる
use_col_fig = [["col_num","row_num"],["amount_of_data", "use_data"]]
name_fig = [["number of columns", "number of rows"],["number of data","data name"]]
for p in range(plt_num[0]):
    for q in range(plt_num[1]):
        
        
        
        for i in range(len_models):
            ax[p,q].scatter(results_model[i][use_col_fig[p][q]], results_model[i]["size"], label=models_name[i], \
                    marker=marker_list[i], c=c_list[i])

        ax[p,q].set_title("{} - data size ".format(name_fig[p][q]))
        ax[p,q].set_xlabel("{}".format(name_fig[p][q]))
        ax[p,q].set_ylabel("data size(byte)")
        if p==0 and q==0:
            ax[p,q].legend(bbox_to_anchor=(-0.3, 1), loc="upper left",borderaxespad=0)
        # use_dataの時、x軸のラベルを回転
        if p==1 and q==1:
            ax[p,q].tick_params(axis='x', labelrotation=90)

#plt.show()
plt.savefig(fig_path+"data_size.png", format="png",dpi=dpi)



# 機械学習スコアの表

# 機械学習スコアの表
# r2 score
marker_list = ["o","o","o","o"]
marker_list2 = ["x","x","x","x"]
c_list = ["#0000FF","#FF0000","#FF00FF","#00FF00"]

plt.rcParams["figure.figsize"]=[20,20]

plt_num = (3,2)  # subplots_num
fig, ax = plt.subplots(plt_num[0],plt_num[1])

# figごとにまとめる
use_col_fig = [["col_num","row_num"],["amount_of_data", "fit_time"],["use_data","use_data"]]
name_fig = [["number of columns", "number of rows"],["number of data","fitting time"],["use data", "use data"]]
for p in range(plt_num[0]):
    for q in range(plt_num[1]):

        if p==2 and q==1:
            for i in range(len_models):
                ax[p,q].scatter(results_model_r[i][use_col_fig[p][q]], results_model_r[i]["diff_Logistic/Linear"], \
                label=models_name[i]+"_Linear", \
                    marker=marker_list[i],c=c_list[i])
                ax[p,q].scatter(results_model_r[i][use_col_fig[p][q]], results_model_r[i]["diff_GradientBoosting"], \
                    label=models_name[i]+"_Gradient", \
                        marker=marker_list2[i], c=c_list[i])
        else: 
            for i in range(len_models):
                ax[p,q].scatter(results_model_r[i][use_col_fig[p][q]], results_model_r[i]["new_r2/accuracy_Logistic/Linear"], \
                    label=models_name[i]+"_Linear", \
                        marker=marker_list[i],c=c_list[i])
                ax[p,q].scatter(results_model_r[i][use_col_fig[p][q]], results_model_r[i]["new_r2/accuracy_GradientBoosting"], \
                    label=models_name[i]+"_Gradient", \
                        marker=marker_list2[i], c=c_list[i])

            ax[p,q].scatter(results_model_r[0][use_col_fig[p][q]], results_model_r[0]["raw_r2/accuracy_Logistic/Linear"], label="raw_data_Random", \
            marker="d", c="#000000")
            ax[p,q].scatter(results_model_r[0][use_col_fig[p][q]], results_model_r[0]["raw_r2/accuracy_GradientBoosting"], label="raw_data_Gradient", \
            marker="*", c="#000000")

        ax[p,q].set_title("{} - r2_score".format(name_fig[p][q]))
        ax[p,q].set_xlabel("{}".format(name_fig[p][q]))
        if p==2 and q==1:
            ax[p,q].set_ylabel("new r2 score - raw r2 score")
        else:
            ax[p,q].set_ylabel("r2 score")
        if p==0 and q==0:
            ax[p,q].legend(bbox_to_anchor=(-0.35, 1), loc="upper left",borderaxespad=0)
        # use_dataの時、x軸のラベルを回転
        if p==2 and (q==0 or q==1):
            ax[p,q].tick_params(axis='x', labelrotation=90)


#plt.rcParams["figure.figsize"]=[20,10]
#plt.show()
plt.savefig(fig_path+"r2_score.png", format="png",dpi=dpi)


# 機械学習スコアの表
# accuracy score
marker_list = ["o","o","o","o"]
marker_list2 = ["x","x","x","x"]
c_list = ["#0000FF","#FF0000","#FF00FF","#00FF00"]

plt.rcParams["figure.figsize"]=[20,20]

plt_num = (3,2)  # subplots_num
fig, ax = plt.subplots(plt_num[0],plt_num[1])

# figごとにまとめる
use_col_fig = [["col_num","row_num"],["amount_of_data", "fit_time"],["use_data","use_data"]]
name_fig = [["number of columns", "number of rows"],["number of data","fitting time"],["use data", "use data"]]
for p in range(plt_num[0]):
    for q in range(plt_num[1]):

        if p==2 and q==1:
            for i in range(len_models):
                ax[p,q].scatter(results_model_c[i][use_col_fig[p][q]], results_model_c[i]["diff_Logistic/Linear"], \
                label=models_name[i]+"_Logistic", \
                    marker=marker_list[i],c=c_list[i])
                ax[p,q].scatter(results_model_c[i][use_col_fig[p][q]], results_model_c[i]["diff_GradientBoosting"], \
                    label=models_name[i]+"_Gradient", \
                        marker=marker_list2[i], c=c_list[i])
        else: 
            for i in range(len_models):
                ax[p,q].scatter(results_model_c[i][use_col_fig[p][q]], results_model_c[i]["new_r2/accuracy_Logistic/Linear"], \
                    label=models_name[i]+"_Logistic", \
                        marker=marker_list[i],c=c_list[i])
                ax[p,q].scatter(results_model_c[i][use_col_fig[p][q]], results_model_c[i]["new_r2/accuracy_GradientBoosting"], \
                    label=models_name[i]+"_Gradient", \
                        marker=marker_list2[i], c=c_list[i])

            ax[p,q].scatter(results_model_c[0][use_col_fig[p][q]], results_model_c[0]["raw_r2/accuracy_Logistic/Linear"], label="raw_data_Random", \
            marker="d", c="#000000")
            ax[p,q].scatter(results_model_c[0][use_col_fig[p][q]], results_model_c[0]["raw_r2/accuracy_GradientBoosting"], label="raw_data_Gradient", \
            marker="*", c="#000000")

        ax[p,q].set_title("{} - accuracy_score".format(name_fig[p][q]))
        ax[p,q].set_xlabel("{}".format(name_fig[p][q]))
        if p==2 and q==1:
            ax[p,q].set_ylabel("new accuracy score - raw accuuracy score")
        else:
            ax[p,q].set_ylabel("r2 score")
        if p==0 and q==0:
            ax[p,q].legend(bbox_to_anchor=(-0.35, 1), loc="upper left",borderaxespad=0)
        # use_dataの時、x軸のラベルを回転
        if p==2 and (q==0 or q==1):
            ax[p,q].tick_params(axis='x', labelrotation=90)


#plt.rcParams["figure.figsize"]=[20,10]
#plt.show()
plt.savefig(fig_path+"accuracy_score.png", format="png",dpi=dpi)



# 各スコアの表

if score:
    marker_list = ["o","o","o","o"]
    marker_list2 = ["x","x","x","x"]
    c_list = ["#0000FF","#FF0000","#FF00FF","#00FF00"]

    # figごとにまとめる
    use_col_list = ["scores_mean","LogisticDetection", "SVCDetection", "GMLogLikelihood","KSTest","KSTestExtended","ContinuousKLDivergence"]


    for p in range(len(use_col_list)):
        for i in range(len_models):
            results_row = results_model[i].shape[0]
            if p==0:
                lavel = models_name[i]
            else:
                lavel = None

            plt.scatter(np.array([use_col_list[p] for x in range(results_row)]), results_model[i][use_col_list[p]],
                    label=lavel, \
                        marker=marker_list[i],c=c_list[i])

    plt.title("score name - score")
    plt.xlabel("score name")
    plt.ylabel("score")
    plt.legend(bbox_to_anchor=(-0.15, 1), loc="upper left",borderaxespad=0)


    plt.rcParams["figure.figsize"]=[20,10]
    #plt.show()
    plt.savefig(fig_path+"many_score.png", format="png",dpi=dpi)


    results.dropna(how="all",axis=1).to_csv(results_path+"dropna.csv", index=False)

