###################################################
# Rating Products
###################################################

# - Average
# - Time-Based Weighted Average
# - User-Based Weighted Average
# - Weighted Rating


############################################
# Uygulama: Kullanıcı ve Zaman Ağırlıklı Kurs Puanı Hesaplama
############################################

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# (50+ Saat) Python A-Z™: Veri Bilimi ve Machine Learning
# Puan: 4.8 (4.764925)
# Toplam Puan: 4611
# Puan Yüzdeleri: 75, 20, 4, 1, <1
# Yaklaşık Sayısal Karşılıkları: 3458, 922, 184, 46, 6

df = pd.read_csv("HAFTA 4/rating_products/course_reviews.csv")
df.head()
#   Rating            Timestamp             Enrolled  Progress  Questions Asked  Questions Answered
#0 5.00000  2021-02-05 07:45:55  2021-01-25 15:12:08   5.00000          0.00000             0.00000
#1 5.00000  2021-02-04 21:05:32  2021-02-04 20:43:40   1.00000          0.00000             0.00000
#2 4.50000  2021-02-04 20:34:03  2019-07-04 23:23:27   1.00000          0.00000             0.00000
#3 5.00000  2021-02-04 16:56:28  2021-02-04 14:41:29  10.00000          0.00000             0.00000
#4 4.00000  2021-02-04 15:00:24  2020-10-13 03:10:07  10.00000          0.00000             0.00000
df.shape #Bakalım burada kaç tane değerlendirme var?
#(4323, 6)


# rating dagılımı
df["Rating"].value_counts() #Puanların dağılım bilgisine erişmek için
#5.00000    3267 #hangi puandan kaç adet
#4.50000     475
#4.00000     383
#3.50000      96
#3.00000      62
#1.00000      15
#2.00000      12
#2.50000      11
#1.50000       2
#Name: Rating, dtype: int64



df["Questions Asked"].value_counts() #sorulan sorular ilgili bir bilgi alalım
#0.00000     3867 #hiç soru sormayan kaç kişi
#1.00000      276
#2.00000       80 #bu 80 kişinin (ort 2 soru soran) kaç puan verdiğini merak edersek eğer (aşağıda)
#3.00000       43
#4.00000       15
#5.00000       13
#6.00000        9
#8.00000        5
#9.00000        3
#14.00000       2
#11.00000       2 #11 soru soran kaç kişi
#7.00000        2
#10.00000       2
#15.00000       2
#22.00000       1
#12.00000       1
#Name: Questions Asked, dtype: int64



df.groupby("Questions Asked").agg({"Questions Asked": "count",
                                   "Rating": "mean"})
#                 Questions Asked  Rating
#Questions Asked
#0.00000                     3867 4.76519
#1.00000                      276 4.74094
#2.00000                       80 4.80625
#3.00000                       43 4.74419
#4.00000                       15 4.83333
#5.00000                       13 4.65385
#6.00000                        9 5.00000
#7.00000                        2 4.75000
#8.00000                        5 4.90000
#9.00000                        3 5.00000
#10.00000                       2 5.00000
#11.00000                       2 5.00000
#12.00000                       1 5.00000
#14.00000                       2 4.50000
#15.00000                       2 3.00000
#22.00000                       1 5.00000


df.head()
#asıl amacımız: bu kursa verilen puanların puanını hesaplamak

####################
# Average
####################

# Ortalama Puan
df["Rating"].mean()
#4.764284061993986
#hiçbir şey bilmiyormuşçasına puanlar burada ort.sını hesaplayalım dersek
#bütün kurslar için aynı ortalama hesabı yapmamız durumunda yanlılık ortaya çıkabilir.



####################
# Time-Based Weighted Average
####################
# Puan Zamanlarına Göre Ağırlıklı Ortalama

df.head()
df.info()
#<class 'pandas.core.frame.DataFrame'>
#RangeIndex: 4323 entries, 0 to 4322
#Data columns (total 6 columns):
# #   Column              Non-Null Count  Dtype
#---  ------              --------------  -----
# 0   Rating              4323 non-null   float64
# 1   Timestamp           4323 non-null   object
# 2   Enrolled            4323 non-null   object
# 3   Progress            4323 non-null   float64
# 4   Questions Asked     4323 non-null   float64
# 5   Questions Answered  4323 non-null   float64
#dtypes: float64(4), object(2)
#memory usage: 202.8+ KB



df["Timestamp"] = pd.to_datetime(df["Timestamp"]) #Timestamp değişkeni object türünde, bunu zaman değişkenine çevirmek istiyoruz.

#yapılan tüm yorumları gün cinsinden ifade etmemiz gerekiyor. 30 gün önce gibi.
current_date = pd.to_datetime('2021-02-10 0:0:0') #current date adıyla bir değişken oluşturuyorum bir string değer veriyrum ve bunu tarihe çevirmesini istiyorum.

df["days"] = (current_date - df["Timestamp"]).dt.days #bugünün tarihinden tüm yorumarın tarihini çıkaralım ve bunu gün cinsinden ifade edelim.
#df["days"], yeni değişken oluşturalım.

#bu veri setinde son 30 günde yapılan yorumlara nasıl erişirim?
df.loc[df["days"] <= 30, "Rating"].mean()
#4.775773195876289
df[df["days"] <= 30].count()
#Rating                194
#Timestamp             194
#Enrolled              194
#Progress              194
#Questions Asked       194
#Questions Answered    194
#days                  194
#dtype: int64

df.loc[df["days"] > 30, "Rating"] #son 30 gündeki Rate'ler geldi
df.loc[df["days"] > 30, "Rating"] .mean()
# 4.763744248001937

#30 günden fazla ve 90 günden az olan günlere bakıyoruz
df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean()
# 4.763833992094861

#eski tarihli günlere bamak istersek eğer:
df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean()
#4.752503576537912

#demekki son zamanlarda kursun memnuniyeti ile ilgili bir artış var.
df.loc[(df["days"] > 180), "Rating"].mean()
#4.76641586867305

#bunları ağırlıklandırarak rate hesaplayalım, zamana göre ağırlıklı ortalamayı hesaplayalım: toplamı 100 olmalı, \ ifadesi aşağı geçmek içindir.
df.loc[df["days"] <= 30, "Rating"].mean() * 28/100 + \
    df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean() * 26/100 + \
    df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean() * 24/100 + \
    df.loc[(df["days"] > 180), "Rating"].mean() * 22/100
# 4.765025682267194


def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["days"] <= 30, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["days"] > 30) & (dataframe["days"] <= 90), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["days"] > 90) & (dataframe["days"] <= 180), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["days"] > 180), "Rating"].mean() * w4 / 100

time_based_weighted_average(df)
#4.765025682267194

time_based_weighted_average(df, 30, 26, 22, 22)
# 4.765491074653962

#artık zamana göre ağırlıklandırmayı yapabiliyoruz.
#herkesin verdiği ağırlık aynı olmalı dersek ne olur?


####################
# User-Based Weighted Average (kullanıcı temelli ağırlıklı ortalama)
####################

#user based=user quality


#acaba kursu izleme oranlarına göre daha farklı bir ağırlık mı olmalı?
df.head()

df.groupby("Progress").agg({"Rating": "mean"}) #farklı izleme oranlarında (ilerlemelerinde=progress) farklı puanlar var (rating)
#kursun izlenmesiyle doğru orantılı bir şekilde artan ağırlıklandırma ile puanın hesaplanmasını istiyoruz.

df.loc[df["Progress"] <= 10, "Rating"].mean() * 22 / 100 + \
    df.loc[(df["Progress"] > 10) & (df["Progress"] <= 45), "Rating"].mean() * 24 / 100 + \
    df.loc[(df["Progress"] > 45) & (df["Progress"] <= 75), "Rating"].mean() * 26 / 100 + \
    df.loc[(df["Progress"] > 75), "Rating"].mean() * 28 / 100
#4.800257704672543



def user_based_weighted_average(dataframe, w1=22, w2=24, w3=26, w4=28):
    return dataframe.loc[dataframe["Progress"] <= 10, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 10) & (dataframe["Progress"] <= 45), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 45) & (dataframe["Progress"] <= 75), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 75), "Rating"].mean() * w4 / 100


user_based_weighted_average(df, 20, 24, 26, 30)
#4.803286469062915

####################
# Weighted Rating (time based + user base) bir araya getirilerek ağırlıklandırma çalışması
####################

def course_weighted_rating(dataframe, time_w=50, user_w=50):
    return time_based_weighted_average(dataframe) * time_w/100 + user_based_weighted_average(dataframe)*user_w/100

course_weighted_rating(df)
#4.782641693469868

course_weighted_rating(df, time_w=40, user_w=60)
#4.786164895710403









