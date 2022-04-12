# --------------------------
# (c) yuki nagae @ tryeting
# MIT
# --------------------------
# 3週間分のデータを使って、次の5日間の需要を予測してみよう

# ///////////////////////////////////////////////////////
# 改造編
# 引数受け取りで、スタンドアローン化
# Rscript --vanilla --slave <Rファイル> <引数1> で実行
# args1 <- commandArgs(trailingOnly=TRUE)[1]


# データ取得... 日本語ファイルのため、いきなり読めない。
tmpfile <-  tempfile()
system(paste("curl http://www.jepx.org/market/excel/spot_2018.csv > ", tmpfile, sep=""))
system(paste("nkf -w --overwrite ", tmpfile, sep=""))  # nkfによる文字エンコーディングの変換
d <- read.csv(tmpfile)  # UTF8によるデータ読み込み

# ///////////////////////////////////////////////////////
# データ取得... 例えばこれを関数にする(LIVE)
tmpfile <-  tempfile()
system(paste("curl http://www.jepx.org/market/excel/spot_2018.csv > ", tmpfile, sep=""))
system(paste("nkf -w --overwrite ", tmpfile, sep=""))  # nkfによる文字エンコーディングの変換
d <- read.csv(tmpfile)  # UTF8によるデータ読み込み
# ///////////////////////////////////////////////////////


# 方針(1)〜5日前のデータを使用して(エリアプライス関西)を予測してみよう！
OBJ <- d[,12]

# ///////////////////////////////////////////////////////
# 例えばこれをより複雑な関数にする
y <- embed(OBJ, 5)[,5]
X <- embed(OBJ, 5)[,1]
D <- data.frame("y"=y, "X"=X)
# ///////////////////////////////////////////////////////

# 入力データを吐き出すための一時ファイル作り
inputf <- tempfile()

# 一時ファイルにcsv吐き出し
write.csv(D, inputf, quote=F, row.names=F)

# system関数でデータをpythonに流し込み...出力を一時ファイルへ
system(paste(

    "python ./4_standalone.py ", inputf
    
    ,sep=""))

# 結果を受け取り、Rからカレントディレクトリにcsvとして焼き直し
R2 <- read.csv("./R2SCORE.csv")
message(paste("R2SCORE : ", R2, sep=""))