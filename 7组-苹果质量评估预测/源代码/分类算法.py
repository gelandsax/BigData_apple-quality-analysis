import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as mfm
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, accuracy_score, precision_score, \
    recall_score, f1_score
import seaborn as sns
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_predict
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

font_path = r'C:\Windows\Fonts\msyh.ttc'
prop = mfm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()

warnings.filterwarnings("ignore")

filename='apple_quality.csv'
data=pd.read_csv(filename,header=0)
df=pd.DataFrame(data)

array=df.values
X=array[:,1:8]
Y=array[:,8]

seed=41
num_folds = 5

X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.4, random_state=seed)
X_validation, X_test, Y_validation, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=seed)

kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
#KNN
model_KNN=KNeighborsClassifier(n_neighbors=5)
model_KNN.fit(X_train,Y_train)
probas=model_KNN.predict_proba(X_test)[:,1]

true_labels = Y_test

# 计算 ROC 曲线的值
fpr, tpr, thresholds = roc_curve(true_labels, probas)

# 计算 AUC 值
roc_auc = auc(fpr, tpr)

# 绘制 ROC 曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('KNN_ROC')
plt.legend(loc="lower right")
plt.show()

#参数优化
param_dist={'n_neighbors':[3,5,7],
            'weights':['uniform','distance'],
            'algorithm':['auto','ball_tree','kd_tree','brute']}
random_search = RandomizedSearchCV(
estimator=KNeighborsClassifier(),
    param_distributions=param_dist,
    n_iter=100,  # 迭代次数
    scoring='accuracy',  # 评价指标
    cv=kfold,  # 交叉验证策略
    random_state=seed,
    n_jobs=-1  # 并行处理
)
random_search.fit(X_train,Y_train)
# 输出最佳参数和得分
print("Best Parameters:", random_search.best_params_)
print("Best Accuracy:", random_search.best_score_)
#优化后的模型
model_KNN=KNeighborsClassifier(n_neighbors=7,weights='distance',algorithm='auto')
model_KNN.fit(X_train,Y_train)
probas=model_KNN.predict_proba(X_test)[:,1]

true_labels = Y_test

# 计算 ROC 曲线的值
fpr, tpr, thresholds = roc_curve(true_labels, probas)

# 计算 AUC 值
roc_auc = auc(fpr, tpr)

# 绘制 ROC 曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('KNN_ROC_better')
plt.legend(loc="lower right")
plt.show()
#混校矩阵
prediction = model_KNN.predict(X_test)
matrix = confusion_matrix(Y_test,prediction)
classes=['0','1']
dataframe = pd.DataFrame(data=matrix,index=classes,columns=classes)
print(dataframe)
report = classification_report(Y_test, prediction)
print(report)
plt.figure(figsize=(8, 6))

sns.heatmap(dataframe, annot=True, cmap='Blues', fmt='d',square=True,
            annot_kws={"size": 14}, linewidths=0.5, linecolor='black')

plt.title('混淆矩阵热力图')
plt.xlabel('预测类别')
plt.ylabel('真实类别')
plt.show()

#使用验证集评估模型效果
predictions=model_KNN.predict(X_validation)
accuracy = accuracy_score(Y_validation, predictions)
precision = precision_score(Y_validation, predictions)
recall = recall_score(Y_validation, predictions)
f1 = f1_score(Y_validation, predictions)
# 输出评估结果
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
#SVC--------------------------------------------------------------
model_SVC=SVC(probability=True)
model_SVC.fit(X_train,Y_train)

probas = model_SVC.predict_proba(X_test)[:, 1]

# 计算 ROC 曲线的值
fpr, tpr, thresholds = roc_curve(Y_test, probas)

# 计算 AUC 值
roc_auc = auc(fpr, tpr)

# 绘制 ROC 曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVC_ROC')
plt.legend(loc="lower right")
plt.show()

#参数优化
param_grid={
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}
grid_search=GridSearchCV(estimator=model_SVC,param_grid=param_grid,cv=num_folds)
grid_search.fit(X_train,Y_train)
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

#优化后的模型
model_SVC=SVC(C=1,gamma='auto',kernel='rbf',probability=True)
model_SVC.fit(X_train,Y_train)

probas = model_SVC.predict_proba(X_test)[:, 1]

# 计算 ROC 曲线的值
fpr, tpr, thresholds = roc_curve(Y_test, probas)

# 计算 AUC 值
roc_auc = auc(fpr, tpr)

# 绘制 ROC 曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVC_ROC_better')
plt.legend(loc="lower right")
plt.show()

prediction = model_SVC.predict(X_test)
matrix = confusion_matrix(Y_test,prediction)
classes=['0','1']
dataframe = pd.DataFrame(data=matrix,index=classes,columns=classes)
print(dataframe)
report = classification_report(Y_test, prediction)
print(report)
plt.figure(figsize=(8, 6))

sns.heatmap(dataframe, annot=True, cmap='Blues', fmt='d',square=True,
            annot_kws={"size": 14}, linewidths=0.5, linecolor='black')

plt.title('混淆矩阵热力图')
plt.xlabel('预测类别')
plt.ylabel('真实类别')
plt.show()

#使用验证集评估模型效果
predictions=model_SVC.predict(X_validation)
accuracy = accuracy_score(Y_validation, predictions)
precision = precision_score(Y_validation, predictions)
recall = recall_score(Y_validation, predictions)
f1 = f1_score(Y_validation, predictions)
# 输出评估结果
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

#RFC-----------------------------------------------------------------
model_RFC=RandomForestClassifier(n_estimators=500)

model_RFC.fit(X_train, Y_train)

# 获取测试集上的预测概率
probas = model_RFC.predict_proba(X_test)[:, 1]

# 计算 ROC 曲线的值
fpr, tpr, thresholds = roc_curve(Y_test, probas)

# 计算 AUC 值
roc_auc = auc(fpr, tpr)

# 绘制 ROC 曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('RFC_ROC')
plt.legend(loc="lower right")
plt.show()

#参数优化

param_dist = {
    'n_estimators': np.arange(50, 151, 10),
    'max_features': np.arange(2, 11),
    'max_depth': [None] + list(np.arange(5, 26, 5)),
    'min_samples_split': np.arange(2, 11),
    'min_samples_leaf': np.arange(1, 11),
    'bootstrap': [True, False]
}
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=seed),
    param_distributions=param_dist,
    n_iter=20,  # 迭代次数
    scoring='accuracy',  # 评价指标
    cv=kfold,  # 交叉验证策略
    random_state=seed,
    n_jobs=-1  # 并行处理
)

# 执行随机搜索
random_search.fit(X_train, Y_train)

# 输出最佳参数和得分
print("Best Parameters:", random_search.best_params_)
print("Best Accuracy:", random_search.best_score_)

best_params = random_search.best_params_
model_RFC=RandomForestClassifier(**best_params,random_state=seed)
model_RFC.fit(X_train, Y_train)

# 获取测试集上的预测概率
probas = model_RFC.predict_proba(X_test)[:, 1]

# 计算 ROC 曲线的值
fpr, tpr, thresholds = roc_curve(Y_test, probas)

# 计算 AUC 值
roc_auc = auc(fpr, tpr)

# 绘制 ROC 曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('RFC_ROC_better')
plt.legend(loc="lower right")
plt.show()

prediction = model_RFC.predict(X_test)
matrix = confusion_matrix(Y_test,prediction)
classes=['0','1']
dataframe = pd.DataFrame(data=matrix,index=classes,columns=classes)
print(dataframe)
report = classification_report(Y_test, prediction)
print(report)
plt.figure(figsize=(8, 6))

sns.heatmap(dataframe, annot=True, cmap='Blues', fmt='d',square=True,
            annot_kws={"size": 14}, linewidths=0.5, linecolor='black')

plt.title('混淆矩阵热力图')
plt.xlabel('预测类别')
plt.ylabel('真实类别')
plt.show()

#使用验证集评估模型效果
predictions=model_RFC.predict(X_validation)
accuracy = accuracy_score(Y_validation, predictions)
precision = precision_score(Y_validation, predictions)
recall = recall_score(Y_validation, predictions)
f1 = f1_score(Y_validation, predictions)
# 输出评估结果
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)