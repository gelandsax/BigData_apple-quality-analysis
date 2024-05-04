# 开发时间：2024/4/6
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as mfm
from scipy.stats import spearmanr
import seaborn as sns
import warnings

font_path = r'C:\Windows\Fonts\msyh.ttc'
prop = mfm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()

warnings.filterwarnings("ignore")

filename='apple_quality.csv'
data=pd.read_csv(filename,header=0)
columns=data.columns
print(columns)
print(data.shape)
df=pd.DataFrame(data)

#作图

#好苹果和坏苹果的分布
Quality_counts = df['Quality'].value_counts()

good_count=Quality_counts[1]
bad_count=Quality_counts[0]

sum=good_count+bad_count

labels=['good','bad']
colors=['green','yellow']
sizes=[good_count/sum*100,bad_count/sum*100]
explodes=(0.1,0)

plt.pie(sizes,explode=explodes,labels=labels,colors=colors,
        autopct="%1.1f%%",shadow=True)
plt.title('苹果好坏占比')
plt.show()

#Sperman 相关系数矩阵和P值矩阵
spearman_corr_matrix, p_value_matrix = spearmanr(data.iloc[:, 1:], axis=1)

print("Spearman 相关系数矩阵:")
print(spearman_corr_matrix)
df_spearman=pd.DataFrame(spearman_corr_matrix)
print("对应的 p 值矩阵:")
print(p_value_matrix)
df_P=pd.DataFrame(p_value_matrix)
# with pd.ExcelWriter('Spearman相关系数以及P值矩阵.xlsx') as writer:
#     # 将系数矩阵写入 sheet1
#     df_spearman.to_excel(writer, sheet_name='Sheet1', index=True)
#
#     # 将P值矩阵写入 sheet2
#     df_P.to_excel(writer, sheet_name='Sheet2', index=True)

#相关系数热力图
names=data.columns[1:]
correlations = data.corr()
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,8,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names, rotation=90, ha='right')
ax.set_yticklabels(names)
for tick in ax.get_xticklabels():
    tick.set_ha('right')
plt.show()