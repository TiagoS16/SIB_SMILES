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


def metrics(label_test, predict):
    print('METRICS:')
    print(f"Accuracy score:\n{accuracy_score(label_test, predict)}\n")
    print(f"Recall score:\n{recall_score(label_test, predict, average='weighted')}\n")
    print(f"Precison score:\n{precision_score(label_test, predict, average='weighted', zero_division=0)}\n")
    print(f"F1-score:\n{f1_score(label_test, predict, average='weighted')}\n")
    print(f"MCC score:\n{matthews_corrcoef(label_test, predict)}\n")
    print(f"Confusion matrix:\n{confusion_matrix(label_test, predict)}\n")
    print(f"Classification report:\n{classification_report(label_test, predict, zero_division=True)}\n")
    # ConfusionMatrixDisplay.from_predictions(label_test, predict)
    # plt.show()


def ml(model, dataset, labels, param=None, rand=None):
    # SPLIT
    data_train, data_test, label_train, label_test = train_test_split(dataset, labels, test_size=0.3)

    # k-fold
    kfold = StratifiedKFold(n_splits=5, random_state=rand, shuffle=False)
    # Cross validation
    scores_scoring = cross_val_score(model, X=data_train, y=label_train, cv=kfold, scoring='accuracy')
    print(f'Cross Validation accuracy score: {np.mean(scores_scoring)}\n')

    # model training - FIT
    model.fit(data_train, label_train)

    # PREDICT
    predict = model.predict(X=data_test)
    base_model = accuracy_score(label_test, predict)
    print('Base Model Accuracy: {:.3f}\n'.format(base_model))

    if param != None:
        ## OPTIMIZATION
        search = HalvingGridSearchCV(estimator=model, param_grid=param, cv=kfold, scoring='accuracy', random_state=rand,
                                     n_jobs=-1)
        # search = HalvingRandomSearchCV(estimator=model, param_distributions=param, cv=kfold, scoring='accuracy',
        #                                random_state=rand, n_jobs=-1)
        search.fit(X=data_train, y=label_train)

        best_params = search.best_params_
        print(f'{best_params}\n')
        # OPTI MODEL FITTED
        best_model = search.best_estimator_
        predict_opt = best_model.predict(X=data_test)
        opt_model = accuracy_score(label_test, predict_opt)
        print('Optimized Model Accuracy: {:.3f}\n'.format(opt_model))

        improv = ((opt_model-base_model)/base_model*100)
        print('Optimized model improved {:.3f}% over base model.\n'.format(improv))

        if improv >= 0:
            predict = predict_opt

    # Metrics
    metrics(label_test, predict)


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
nn = MLPClassifier(early_stopping=True)
models = [rf, nb, knn, voting, nn]

# PARAMETERS
params_rf = {'n_estimators': range(10, 211, 50), 'criterion': ['entropy', 'gini'], 'max_features': ['sqrt', 'log2', None],
          'bootstrap': [True, False]}
params_knn = {'n_neighbors': range(2, 11, 2), 'weights': ['distance', 'uniform'], 'leaf_size': range(10, 50, 10), 'p': [1, 2]}
params_nn = {'activation': ['identity', 'logistic', 'tanh', 'relu'], 'learning_rate': ['constant', 'invscaling', 'adaptive']}
params = [params_rf, None, params_knn, None, params_nn]

# Descriptors
print('DESCRIPTORS')
for i in range(len(models)):
    print(models[i])
    ml(models[i], descriptors_data, descriptors_label, params[i])


# Fingerprints
print('FINGERPRINTS')
for i in range(len(models)):
    print(models[i])
    ml(models[i], fingerprint_data, fingerprint_label, params[i])
