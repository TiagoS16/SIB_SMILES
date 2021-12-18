import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, f1_score, recall_score,
                             matthews_corrcoef, ConfusionMatrixDisplay, precision_recall_curve, PrecisionRecallDisplay,
                             average_precision_score, auc, RocCurveDisplay, roc_curve, precision_score, roc_auc_score)
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV, HalvingGridSearchCV


rand = random.randrange(2147483647)
print(rand)


def metrics(label_test, predict, predict_prob=None, n_label=None):
    print(f"Accuracy score:\n{accuracy_score(label_test, predict)}\n")
    print(f"Recall score:\n{recall_score(label_test, predict, average='weighted')}\n")
    print(f"Precison score:\n{precision_score(label_test, predict, average='weighted', zero_division=0)}\n")
    print(f"F1-score:\n{f1_score(label_test, predict, average='weighted')}\n")
    # print(f"ROC AUC score:\n{roc_auc_score(label_test, predict_prob[:, 1], average='weighted')}\n")
    print(f"MCC score:\n{matthews_corrcoef(label_test, predict)}\n")
    print(f"Confusion matrix:\n{confusion_matrix(label_test, predict)}\n")
    print(f"Classification report:\n{classification_report(label_test, predict, zero_division=True)}\n")
    # ConfusionMatrixDisplay.from_predictions(label_test, predict)
    # plt.show()


def ml(model, dataset, labels, rand=None):
    # SPLIT
    data_train, data_test, label_train, label_test = train_test_split(dataset, labels, test_size=0.3)

    # k-fold
    kfold = StratifiedKFold(n_splits=5, random_state=rand, shuffle=True)
    # Cross validation
    scores_scoring = cross_val_score(model, X=data_train, y=label_train, cv=kfold, scoring='accuracy')
    print(f'Cross Validation accuracy score: {np.mean(scores_scoring)}')

    # model training - FIT
    model.fit(data_train, label_train)

    # PREDICT
    predict = model.predict(X=data_test)
    predict_prob = model.predict_proba(X=data_test)

    # Metrics
    metrics(label_test, predict, predict_prob)


## BINARY ##
# LOAD DATA
descriptors = pd.read_csv('../dataset/binary_class/descriptors_fs.csv', sep=',')
fingerprint = pd.read_csv('../dataset/binary_class/rdk_fs.csv', sep=',')

descriptors_data = descriptors.drop("activity", axis=1)
descriptors_label = descriptors["activity"]
fingerprint_data = fingerprint.drop("activity", axis=1)
fingerprint_label = fingerprint["activity"]

# MODELS
rf = RandomForestClassifier(n_jobs=-1)
nb = GaussianNB()
knn = KNeighborsClassifier(n_jobs=-1)
voting = VotingClassifier(estimators=[('Random Forest', rf), ('Naive Bayes', nb), ('KNN', knn)], voting='soft', n_jobs=-1)
nn = MLPClassifier()
models = [rf, nb, knn, voting, nn]

# Descriptors
print('DESCRIPTORS')
for model in models:
    print(model)
    # ml(model, descriptors_data, descriptors_label)

# Fingerprints
print('FINGERPRINTS')
for model in models:
    print(model)
    # ml(model, fingerprint_data, fingerprint_label)


def opti(model, param, rand=None):
    data_train, data_test, label_train, label_test = train_test_split(descriptors_data, descriptors_label, test_size=0.3)

    kfold = StratifiedKFold(n_splits=5, random_state=rand, shuffle=False)
    # search = HalvingGridSearchCV(estimator=model, param_grid=param, cv=kfold, scoring='accuracy', random_state=rand,
    #                              verbose=1, n_jobs=-1)
    search = HalvingRandomSearchCV(estimator=model, param_distributions=param, cv=kfold, scoring='accuracy',
                                   random_state=rand, verbose=1, n_jobs=-1)
    search.fit(X=data_train, y=label_train)

    best_model = search.best_estimator_
    print(best_model)
    predict = best_model.predict(X=data_test)
    metrics(label_test, predict)

params_rf = {'n_estimators': range(10, 211, 50), 'criterion': ['entropy', 'gini'], 'max_features': ['sqrt', 'log2', None],
          'bootstrap': [True, False]}
params_knn = {'n_neighbors': range(2, 11, 2), 'weights': ['distance', 'uniform'], 'leaf_size': range(10, 50, 10), 'p': [1, 2]}
opti(knn, params_knn)

