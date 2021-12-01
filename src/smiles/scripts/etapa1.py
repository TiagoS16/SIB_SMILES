import numpy as np
import pandas as pd
# import os
from scipy import stats
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
from standardizer.CustomStandardizer import CustomStandardizer
from loaders.Loaders import CSVLoader
import sys
sys.path.append('CODE_SIB')

# DIR = os.path.dirname(os.path.realpath('.'))
# file = os.path.join(DIR, 'dataset/TDP1_activity_dataset.csv')


# CARREGAR O DATASET
dataframe = pd.read_csv('../dataset/TDP1_activity_dataset.csv', sep=',')  # definir types para nao dar erros

# dataframe.to_csv(f"{DIR}/dataset/decente.csv")

# ANALISE SIMPLES
print(dataframe.size)
print(dataframe.shape)
print(dataframe.columns)
print(dataframe.dtypes)

print(dataframe.describe())  # todo para cada coluna de valores numericos aka actividades


# PRE PROCESSAMENTO
# NAs
print(dataframe.isna().sum().sum())
print(dataframe.isna().sum())

#Ver o numero de Nan no dataset
msno.bar(dataframe, sort="ascending")
plt.show()

#dataframe = dataframe.dropna(axis=1, how='all')  # da drop de todas as colunas que contenham apenas NAs

# # APAGAR FEATURES ESPECIFICAS
# del dataframe['PUBCHEM_ACTIVITY_URL']  # drop de colunas desnecessarias
# del dataframe['Compound QC']
# # todo questionar sobre colunas CID e SID / apagou algumas colunas das ativiades
# print(dataframe.shape)
# print(dataframe.columns)
#
#
# # ANALISE GRAFICA
# # Pie charts activity_outcome
# activity = dataframe.groupby('PUBCHEM_ACTIVITY_OUTCOME').size()
# labels_activity = dataframe.groupby('PUBCHEM_ACTIVITY_OUTCOME').size().index
# print(dataframe.groupby('PUBCHEM_ACTIVITY_OUTCOME').size())
#
# # Pie charts phenotype
# fenotipo = dataframe.groupby('Phenotype').size()
# labels_fenotipo = dataframe.groupby('Phenotype').size().index
# print(dataframe.groupby('Phenotype').size())
#
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
# ax1.pie(activity, labels=labels_activity, autopct='%1.1f%%', startangle=90)
# ax1.set_title('PUBCHEM_Activity_Outcome')
# ax2.pie(fenotipo, labels=labels_fenotipo, autopct='%1.1f%%', startangle=360)
# ax2.set_title('Phenotype')
# #plt.show()
#
#
# plt.subplots(figsize=(10, 10))
# sns.set(font_scale=1.4)
# plt.title("Activity at 46.23 uM", fontsize=25)
# sns.boxplot(y="Activity at 46.23 uM",
#             data=dataframe, palette="Set3")
# plt.show()
#
# plt.subplots(figsize=(10, 10))
# sns.set(font_scale=1.4)
# plt.title("Activity at 0.00299 uM", fontsize=25)
# sns.boxplot(y="Activity at 0.00299 uM",
#             data=dataframe, palette="Set3")
# plt.show()
#
# plt.subplots(figsize=(10, 10))
# sns.set(font_scale=1.4)
# plt.title("Activity at 0.363 uM", fontsize=25)
# sns.boxplot(y="Activity at 0.363 uM",
#             data=dataframe, palette="Set3")
# plt.show()
#
# plt.subplots(figsize=(10, 10))
# sns.set(font_scale=1.4)
# plt.title("Activity at 9.037 uM", fontsize=25)
# sns.boxplot(y="Activity at 9.037 uM",
#             data=dataframe, palette="Set3")
# plt.show()
#
# plt.subplots(figsize=(10, 10))
# sns.set(font_scale=1.4)
# plt.title("Activity at 1.849 uM", fontsize=25)
# sns.boxplot(y="Activity at 1.849 uM",
#             data=dataframe, palette="Set3")
# plt.show()
#
#
# def standardize(dataset, id_field, mols_field, class_field):
#     loader = CSVLoader(dataset,
#                        id_field=id_field,
#                        mols_field=mols_field,
#                        labels_fields=class_field)
#
#     dataset = loader.create_dataset()
#
#     standardisation_params = {
#         'REMOVE_ISOTOPE': True,
#         'NEUTRALISE_CHARGE': True,
#         'REMOVE_STEREO': False,
#         'KEEP_BIGGEST': True,
#         'ADD_HYDROGEN': False,
#         'KEKULIZE': True,
#         'NEUTRALISE_CHARGE_LATE': True}
#
#     CustomStandardizer(params=standardisation_params).standardize(dataset)
#
#     return dataset
#
#
# dataframe = standardize(file, "PUBCHEM_CID", "smiles", "PUBCHEM_ACTIVITY_OUTCOME")

# BOXPLOT actividades
"""
act1 = dataframe['Activity at 0.0005895491 uM'].dropna()
act2 = dataframe['Activity at 0.00299 uM'].dropna()
act3 = dataframe['Activity at 0.014 uM'].dropna()
act4 = dataframe['Activity at 0.074 uM'].dropna()
act5 = dataframe['Activity at 0.363 uM'].dropna()
act6 = dataframe['Activity at 1.849 uM'].dropna()
act7 = dataframe['Activity at 9.037 uM'].dropna()
act8 = dataframe['Activity at 46.23 uM'].dropna()

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(10, 5))
fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(10, 5))
fig4, (ax7, ax8) = plt.subplots(1, 2, figsize=(10, 5))
ax1.boxplot(act1)
ax1.set_title('Activity at 0.0005895491 uM')
ax2.boxplot(act2)
ax2.set_title('Activity at 0.00299 uM')
ax3.boxplot(act3)
ax3.set_title('Activity at 0.014 uM')
ax4.boxplot(act4)
ax4.set_title('Activity at 0.074 uM')
ax5.boxplot(act5)
ax5.set_title('Activity at 0.363 uM')
ax6.boxplot(act6)
ax6.set_title('Activity at 1.849 uM')
ax7.boxplot(act7)
ax7.set_title('Activity at 9.037 uM')
ax8.boxplot(act8)
ax8.set_title('Activity at 46.23 uM')
"""
# plt.show()




### REUNIAO

# tSNE, Kmeans, PCA
# ativa, inativa
# justificar features que usamos para treinar modelos
# activity outcome - class binaria
# listar as feats que nao tem NAs
# multiclass atraves dos valores de atividades

# limpar todas as colunas que tenham NAs (testar)

# obter features atraves dos SMILES

# selecionar as moleculas ativas e usar a label de potencia como inibidor (e.g.)


# PRE PROCESSAMENTO
# manter tudo para analisar o dataset
# segundo dataset com limpeza de NAs
# terceiro dataset so com moleculas ativas e ver por potencia

# falar de um conjunto de features mais interessantes

