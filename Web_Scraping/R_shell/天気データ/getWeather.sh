#!/bin/bash

# 位置変数を読み込んで、どこの地点の何月のデータを取得するのかを計算
# 大阪府(62)大阪市(47772)の2018年1月の天気を取得するには
# sh ./getWeather.sh 62 47772 2018 1

curl -s "http://www.data.jma.go.jp/obd/stats/etrn/view/daily_s1.php?prec_no=${1}&block_no=${2}&year=${3}&month=${4}&day=1&view=" -H 'Upgrade-Insecure-Requests: 1' -H 'User-Agent: Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Mobile Safari/537.36' -H 'X-DevTools-Emulate-Network-Conditions-Client-Id: 1EC7766F3B796644CAB979D6EB30A5BC' -H "Referer: http://www.data.jma.go.jp/obd/stats/etrn/index.php?prec_no=${1}&block_no=${2}&year=${3}&month=${4}&day=1&view=" --compressed | pup "[class="data_0_0"]" | sed '/td/d' | sed -e 's/--/NA/g' > ./tmp.txt

echo "./tmp.txt"