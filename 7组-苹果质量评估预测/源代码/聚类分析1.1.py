import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn import datasets
import matplotlib.font_manager as mfm
from scipy.stats import spearmanr
import seaborn as sns
import warnings


font_path = r'C:\Windows\Fonts\msyh.ttc'
prop = mfm.FontProperties(fname=font_path)

warnings.filterwarnings("ignore")

filename='apple_quality.csv'
data=pd.read_csv(filename,header=0)
columns=data.columns

X=data.iloc[:,1:4].values
seed = 41
print(columns)
def cluster():
    model = KMeans(n_clusters=3,init='k-means++',random_state=seed,max_iter=300,tol=1e-3)
    model.fit(X)

    predict_y=model.predict(X)
    print(predict_y)

    color_list=['red','blue','yellow']
    point_color = [color_list[i] for i in predict_y]

    fig=plt.figure(figsize=(10,10))
    # ax = Axes3D(fig, rect=[0,0,.95,1],elev=48,azim=134)
    ax = fig.add_subplot(111, projection='3d',elev=48,azim=68)
    ax.scatter(X[:,0],X[:,1],X[:,2],color=point_color,edgecolor="k",s=100)

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    ax.set_xlabel('Size')
    ax.set_ylabel('Weight')
    ax.set_zlabel('Sweetness')
    ax.dist = 10

    for name, label in [('1', 0),
                        ('2', 1),
                        ('3', 2)]:
        ax.text3D(X[predict_y == label, 0].mean(),
                  X[predict_y == label, 1].mean(),
                  X[predict_y == label, 2].mean(), name,
                  horizontalalignment='center',
                  bbox=dict(alpha=.5, edgecolor='w', facecolor='w'),weight='bold')

    plt.show()



if __name__ == '__main__':
    cluster()


