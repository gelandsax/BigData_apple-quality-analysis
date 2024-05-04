import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_predict
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, roc_curve, auc, confusion_matrix
import warnings
import matplotlib.font_manager as mfm
import matplotlib.pyplot as plt

font_path = r'C:\Windows\Fonts\msyh.ttc'
prop = mfm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()
warnings.filterwarnings("ignore")

filename='apple_quality.csv'
data=pd.read_csv(filename,header=0)
df=pd.DataFrame(data)
columns=data.columns
print(columns)
print(data.shape)

array=df.values
X=array[:,1:8]
Y=array[:,8]

seed=41
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.4, random_state=seed)
X_validation, X_test, Y_validation, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=seed)
#模型评估算法（十折交叉检验）
num_folds = 10
scoring = 'neg_mean_squared_error'
models = {}
models['LR'] = LinearRegression()
models['LASSO'] = Lasso()
models['EN'] = ElasticNet()
models['KNN'] = KNeighborsRegressor()
models['CART'] = DecisionTreeRegressor()
models['SVM'] = SVR()

results = []
for key in models:
    kfold = KFold(n_splits=num_folds,random_state=seed,shuffle=True)
    cv_result=cross_val_score(models[key],X_train,Y_train,cv=kfold,scoring=scoring)
    results.append(cv_result)
    print('%s: %f(%f)' % (key, cv_result.mean(), cv_result.std()))

fig = pyplot.figure()
fig.suptitle('Algorithm Comparision')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(models.keys())
pyplot.show()

