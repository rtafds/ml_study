# 位置変数を読み込んで、どこの地点の何月のデータを取得するのかを計算
# 大阪府(62)大阪市(47772)の2018年1月の天気を取得するには
# d <- getWeather(62, 47772, 2018, 1)


getWeather <- function(prec_no, block_no, year, month){

    # シェルの実行
    system(paste(
        "sh ../SRC/getWeather.sh ", prec_no, " ", block_no, " ", year, " ", month
        ,sep=""))

    # データの読み込み
    err <- try(d <- read.csv("./tmp.txt", header=F))
    if(class(err)=="try-error"){
        return(404)
    }
    D <- t(data.frame(matrix(unlist(d), nrow=20)))
    write.csv(D, "./tmp.csv", quote=F, row.names=F)
    D <- read.csv("./tmp.csv")


    # 日付データの作成
    startdate <- as.Date(paste(year,month,1,sep="-"))
    
    if(month==12){
        enddate <- as.Date(paste(year+1,1,1,sep="-")) - 1
    } else {
        enddate <- as.Date(paste(year,month + 1,1,sep="-")) - 1
    }

    D <- data.frame(
        "日付"=head(seq(startdate, enddate, 1), nrow(D)),
        D)

    colnames(D) <- c(
        "日付",
        "平均現地気圧(hPa)",
        "平均海面気圧(hPa)",
        "降水量合計(mm)",
        "降水量最大1時間(mm)",
        "降水量最大10分間(mm)",
        "平均気温(C)",
        "最高気温(C)",
        "最低気温(C)",
        "平均湿度(%)",
        "最小湿度(%)",
        "平均風速(m/s)",
        "最大風速(m/s)",
        "最大風向(m/s)",
        "最大瞬間風速(m/s)",
        "最大瞬間風向(m/s)",
        "日照時間(h)",
        "降雪合計(cm)",
        "最深深雪",
        "天気概況・昼(6-18)",
        "天気概況・夜(18-翌6)")

    write.csv(D, "./tmp.csv", quote=F, row.names=F)
    return(D)
}