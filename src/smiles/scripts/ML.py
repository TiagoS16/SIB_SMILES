#%% md
## IMPORTS
#%%
import sys
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from metrics.metricsFunctions import r2_score, roc_auc_score, precision_score, accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import random
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, f1_score, recall_score,
                             balanced_accuracy_score, fbeta_score, matthews_corrcoef,
                             precision_recall_curve, PrecisionRecallDisplay, average_precision_score, auc,
                             RocCurveDisplay, roc_curve, precision_score, plot_confusion_matrix)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

sys.path.append('src')
#%% md
## LOAD DATASETS
#%%
descriptors = pd.read_csv('../dataset/binary_class/descriptors_fs.csv', sep=',')
fingerprint = pd.read_csv('../dataset/binary_class/rdk_fs.csv', sep=',')

# Activity = pd.read_csv('../dataset/TDP1_activity_dataset.csv', sep=',')
# Activity_Class = copy.deepcopy(Activity)

#%% md
# MACHINE LEARNING
#%%
#Agrupar actividade M46 em classes, sendo que o minimo é da classe é -150 e o máximo é 50

##classe1 [-150, -100[ ##classe2[-100, -50[ ##classe3[-50, 0[  ##classe4 [0, 50]
#
# conditions = [(-150 <= Activity_Class["Activity at 46.23 uM"]) & (Activity_Class["Activity at 46.23 uM"] < -100),
#               (-100 <= Activity_Class["Activity at 46.23 uM"]) & (Activity_Class["Activity at 46.23 uM"] < -50),
#               (-50 <= Activity_Class["Activity at 46.23 uM"]) & (Activity_Class["Activity at 46.23 uM"] < 0),
#               (0 <= Activity_Class["Activity at 46.23 uM"]) & (Activity_Class["Activity at 46.23 uM"] < 50)]
# results = ["0", "1", "2", "3"]
# Activity_Class['Activity at 46.23 uM'] = np.select(conditions, results)
#
# ## realizr pie chart para ver a igualdade das classes, se estão bem distribuidas ou não
# class_activity = Activity_Class.groupby('Activity at 46.23 uM').size()
# print(sum(class_activity))
# print(class_activity)
#
# fig, (ax1) = plt.subplots(1, figsize=(15, 5))
# ax1.pie(class_activity, labels=results, autopct='%1.1f%%', startangle=90)
# ax1.set_title('Activity Classes')
# plt.show()
# #%%


##fazer machine learning :D

def metrics(class_test, cv_predict, cv_predict_prob ):
    precision, recall, thresholds = precision_recall_curve(class_test, cv_predict_prob[:, 1], pos_label=1)
    ap = average_precision_score(class_test, cv_predict_prob[:, 1], pos_label=1)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=ap, estimator_name=None,
                                  pos_label=1)
    disp.plot()

    fpr, tpr, thresholds1 = roc_curve(class_test, cv_predict_prob[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    disp1 = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=None, pos_label=1)
    disp1.plot()

    # print(f"Average precision:\n{average_precision(class_test, cv_predict)}\n")
    print(f"Accuracy score:\n{accuracy_score(class_test, cv_predict)}\n")
    print(f"Balanced accuracy score:\n{balanced_accuracy_score(class_test, cv_predict)}\n")  # average of recall on each class
    # print(f"Top k accuracy score:\n{top_k_accuracy_score(class_test, cv_predict, k=1)}\n")
    print(f"F-beta (2.0) score:\n{fbeta_score(class_test, cv_predict, average='weighted', beta=2.0, pos_label=1)}\n")  # beta > 1 favors recall
    print(f"F1-score (mean):\n{f1_score(class_test, cv_predict, average='weighted')}\n")  # average of precision and recall
    print(f"F1-score (relevant):\n{f1_score(class_test, cv_predict, average='weighted', pos_label=1)}\n")  # relevant label F-score
    print(f"Recall score (mean):\n{recall_score(class_test, cv_predict, average='macro')}\n")  # ability to find all positive samples (jsut for relevant)
    print(f"Recall score (relevant):\n{recall_score(class_test, cv_predict, average='weighted', pos_label=1)}\n")  # ability to find all positive samples (jsut for relevant)
    print(f"Precison score (mean):\n{precision_score(class_test, cv_predict, average='macro', zero_division=0)}\n")
    print(f"Precison score (relevant):\n{precision_score(class_test, cv_predict, average='weighted', pos_label=1, zero_division=0)}\n")
    print(f"Average precision score:\n{ap}\n")
    print(f"PR AUC score:\n{auc(recall, precision)}\n")
    print(f"ROC AUC score:\n{roc_auc}\n")
    print(f"MCC score:\n{matthews_corrcoef(class_test, cv_predict)}\n")  # +1 is perfect, 0 is random, -1 is inverse prediction
    print(f"Confusion matrix:\n{confusion_matrix(class_test, cv_predict)}\n")
    print(f"Classification report:\n{classification_report(class_test, cv_predict, zero_division=True)}\n")  # main classification metrics

rand = random.randrange(2147483647)

##fazer descriptors contra binary_class e comparar métodos de ML
#Data Split
descriptors_data = descriptors.drop("activity", axis=1)
descriptors_label = descriptors["activity"]

data_train, data_test, label_train, label_test = train_test_split(descriptors_data, descriptors_label, test_size=0.3) # 70% training and 30% test

#Scikit-Learn Random Forest
print('Random Forest\n\n')
rf = RandomForestClassifier()

#cross validate model on the full dataset
kfold = StratifiedKFold(n_splits=5, random_state=rand, shuffle=True)
scores_scoring = cross_val_score(rf, X=data_train, y=label_train, cv=kfold, scoring='f1')
print(scores_scoring)

# model training - FIT
rf.fit(data_train, label_train)

# PREDICT
cv_predict = rf.predict(X=data_test)

cv_predict_prob = rf.predict_proba(X=data_test)

#METRICS
metrics(label_test, cv_predict, cv_predict_prob)

#CONFUSION MATRIX
plot_confusion_matrix(rf, data_test, label_test)
plt.title("Random Forest - descriptors")
plt.show()



#Scikit-Learn SVM Classifier
print('SVM\n\n')
svm = SVC(C=1.6, kernel='linear', class_weight='balanced', probability=True, random_state=rand)

#cross validate model on the full dataset
kfold = StratifiedKFold(n_splits=5, random_state=rand, shuffle=True)
scores_scoring = cross_val_score(svm, X=data_train, y=label_train, cv=kfold, scoring='f1')
print(scores_scoring)

# model training
svm.fit(data_train, label_train)

# PREDICT
cv_predict = svm.predict(X=data_test)

cv_predict_prob = svm.predict_proba(X=data_test)

#METRICS
metrics(label_test, cv_predict, cv_predict_prob)

#CONFUSION MATRIX
plot_confusion_matrix(svm, data_test, label_test)
plt.title("SVM - descriptors")
plt.show()





##fazer fingerprints contra binary_class e comparar métodos de ML
#ADICIONAR COLUNA binary_class AOS DATASET fingerprint
print(fingerprint)

#Data Split
fingerprint_data = fingerprint.drop("activity", axis=1)
fingerprint_label = fingerprint["activity"]

data_train, data_test, label_train, label_test = train_test_split(fingerprint_data, fingerprint_label, test_size=0.3) # 70% training and 30% test

#Scikit-Learn Random Forest
print('Random Forest\n\n')
rf = RandomForestClassifier()

#cross validate model on the full dataset
kfold = StratifiedKFold(n_splits=5, random_state=rand, shuffle=True)
scores_scoring = cross_val_score(rf, X=data_train, y=label_train, cv=kfold, scoring='f1')
print(scores_scoring)

# model training - FIT
rf.fit(data_train, label_train)

# PREDICT
cv_predict = rf.predict(X=data_test)

cv_predict_prob = rf.predict_proba(X=data_test)

#METRICS
metrics(label_test, cv_predict, cv_predict_prob)

#CONFUSION MATRIX
plot_confusion_matrix(rf, data_test, label_test)
plt.title("Random Forest - descriptors")
plt.show()



#Scikit-Learn SVM Classifier
print('SVM\n\n')
svm = SVC(C=1.6, kernel='linear', class_weight='balanced', probability=True, random_state=rand)

#cross validate model on the full dataset
kfold = StratifiedKFold(n_splits=5, random_state=rand, shuffle=True)
scores_scoring = cross_val_score(svm, X=data_train, y=label_train, cv=kfold, scoring='f1')
print(scores_scoring)

# model training
svm.fit(data_train, label_train)

# PREDICT
cv_predict = svm.predict(X=data_test)

cv_predict_prob = svm.predict_proba(X=data_test)

#METRICS
metrics(label_test, cv_predict, cv_predict_prob)

#CONFUSION MATRIX
plot_confusion_matrix(svm, data_test, label_test)
plt.title("SVM - descriptors")
plt.show()

# #
# # ##fazer uma conclusão para saber qual o melhor para prever atividade os fungerprints ou os descriptores
# #
# #
# # ##fazer descritores contra classe e comparar métodos de ML
# #descriptors_data já existe
# Activity_Class_compare = copy.deepcopy(Activity_Class)
# Activity_Class_compare.drop(axis=0, index=5182)
# descriptors_class_label = Activity_Class_compare['Activity at 46.23 uM']
#
# data_train, data_test, label_train, label_test = train_test_split(descriptors_data, descriptors_class_label, test_size=0.3) # 70% training and 30% test
#
# #Scikit-Learn Random Forest
# print('Random Forest\n\n')
# random_forest = RandomForestClassifier()
#
# #cross validate model on the full dataset
# kfold = StratifiedKFold(n_splits=10, random_state=rand, shuffle=True)
# scores_scoring = cross_val_score(random_forest, X=data_train, y=label_train, cv=kfold, scoring='f1')
# print(scores_scoring)
#
# # model training - FIT
# random_forest.fit(data_train, label_train)
#
# # PREDICT
# cv_predict = random_forest.predict(X=data_test)
#
# cv_predict_prob = random_forest.predict_proba(X=data_test)
#
# #METRICS
# metrics(label_test, cv_predict, cv_predict_prob)
#
# #CONFUSION MATRIX
# plot_confusion_matrix(random_forest, data_test, label_test)
# plt.title("Random Forest - descriptors")
# plt.show()
#
#
#
# #Scikit-Learn SVM Classifier
# print('SVM\n\n')
# svm = SVC(C=1.6, kernel='linear', class_weight='balanced', probability=True, random_state=rand)
#
# #cross validate model on the full dataset
# kfold = StratifiedKFold(n_splits=10, random_state=rand, shuffle=True)
# scores_scoring = cross_val_score(svm, X=data_train, y=label_train, cv=kfold, scoring='f1')
# print(scores_scoring)
#
# # model training
# svm.fit(data_train, label_train)
#
# # PREDICT
# cv_predict = svm.predict(X=data_test)
#
# cv_predict_prob = svm.predict_proba(X=data_test)
#
# #METRICS
# metrics(label_test, cv_predict, cv_predict_prob)
#
# #CONFUSION MATRIX
# plot_confusion_matrix(svm, data_test, label_test)
# plt.title("SVM - descriptors")
# plt.show()
#
#
#
# #
# # descriptors_compare = copy.deepcopy(descriptors) !TODO perguntar Tiago sobre esta nova label...
# # descriptors_compare.drop(labels="binary_class", axis=1)
# # descriptors_compare["Activity at 46.23 uM"] = Activity_Class_compare["Activity at 46.23 uM"] #TESTAR????
# #
# # ##fazer fingerprints contra classe e comparar métodos de ML !TODO os fingerprints não perdem colunas correto?
#
# fingerprint_label_class = Activity_Class['Activity at 46.23 uM']
#
# data_train, data_test, label_train, label_test = train_test_split(fingerprint_data, fingerprint_label_class, test_size=0.3) # 70% training and 30% test
#
# #Scikit-Learn Random Forest
# print('Random Forest\n\n')
# random_forest = RandomForestClassifier()
#
# #cross validate model on the full dataset
# kfold = StratifiedKFold(n_splits=10, random_state=rand, shuffle=True)
# scores_scoring = cross_val_score(random_forest, X=data_train, y=label_train, cv=kfold, scoring='f1')
# print(scores_scoring)
#
# # model training - FIT
# random_forest.fit(data_train, label_train)
#
# # PREDICT
# cv_predict = random_forest.predict(X=data_test)
#
# cv_predict_prob = random_forest.predict_proba(X=data_test)
#
# #METRICS
# metrics(label_test, cv_predict, cv_predict_prob)
#
# #CONFUSION MATRIX
# plot_confusion_matrix(random_forest, data_test, label_test)
# plt.title("Random Forest - descriptors")
# plt.show()
#
#
#
# #Scikit-Learn SVM Classifier
# print('SVM\n\n')
# svm = SVC(C=1.6, kernel='linear', class_weight='balanced', probability=True, random_state=rand)
#
# #cross validate model on the full dataset
# kfold = StratifiedKFold(n_splits=10, random_state=rand, shuffle=True)
# scores_scoring = cross_val_score(svm, X=data_train, y=label_train, cv=kfold, scoring='f1')
# print(scores_scoring)
#
# # model training
# svm.fit(data_train, label_train)
#
# # PREDICT
# cv_predict = svm.predict(X=data_test)
#
# cv_predict_prob = svm.predict_proba(X=data_test)
#
# #METRICS
# metrics(label_test, cv_predict, cv_predict_prob)
#
# #CONFUSION MATRIX
# plot_confusion_matrix(svm, data_test, label_test)
# plt.title("SVM - descriptors")
# plt.show()
#
#
#
# #
# # ##fazer uma conclusão para saber qual o melhor para prever binário os fingerprints ou os descriptores
# #
# # ##junção de Activity at 46.23 uM a tabela dos descritores, atenção pois houve uma molécula removida dos descritores devido a NA's, por isso atenção na adição da coluna.
#
# Activity.drop(axis=0, index=5182)
# label = Activity['Activity ...']
#
#
#
# #
# # ##fazer então a partir dos descritores a previsão direta da Activity.
# #
