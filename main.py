import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

#データ読み込み
df_balance = pd.read_csv("./data/fy-balance-sheet.csv", header=1)
df_cf = pd.read_csv("./data/fy-cash-flow-statement.csv", header=1)
df_pl = pd.read_csv('./data/fy-profit-and-loss.csv',header=1)
df_dividend = pd.read_csv('./data/fy-stock-dividend.csv',header=1)
df_code = pd.read_csv('./data/data_j.csv')

#企業コードでマージ
df_datas=pd.merge(df_balance, df_cf, on='コード')
df_datas=pd.merge(df_datas, df_pl, on='コード')
df_datas=pd.merge(df_datas, df_dividend, on='コード')
df_datas=pd.merge(df_datas,df_code,on="コード")

#欠損値を0埋め
df_datas.replace("-","0",inplace=True)

#必要なデータだけ取ってくる
params=["総資産","売上高","純資産配当率","短期借入金","長期借入金","自己資本比率","ROE","ROA","EPS"]
df_datastemp=df_datas[params]
df_datastemp.loc[:,:]=scaler.fit_transform(df_datastemp)

temp=[]
for i in range(len(params)):
    temp.append(df_datastemp[params[i]])
np_datas=np.array(temp)
np_datas=np_datas.T

#クラスター数を指定する
clusternum=50

pred=KMeans(n_clusters=clusternum).fit_predict(np_datas)

#グラフにする
df_datas["cluster_id"]=pred
df_datastemp["cluster_id"]=pred

clusterinfo=pd.DataFrame()
for i in range(clusternum):
    clusterinfo[str(i)]=df_datastemp[df_datastemp["cluster_id"]==i].mean()
clusterinfo=clusterinfo.drop("cluster_id")
my_plot = clusterinfo.T.plot(kind='bar', stacked=True, title="Mean Value of 50 Clusters",figsize=(12, 6))
my_plot.set_xticklabels(my_plot.xaxis.get_majorticklabels(), rotation=90)

plt.show()
plt.savefig("bar.jpg")
plt.close('all')

df_datas=df_datas[["cluster_id","コード","銘柄名"]]

df_datas.sort_values("cluster_id",inplace=True)
df_datas.to_csv("output.csv")