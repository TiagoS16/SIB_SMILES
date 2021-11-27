import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

file = '/home/laptop16/Mestrado/2_ano/SIB/Grupo/Code/src/smiles/dataset/TDP1_activity_dataset.csv'
dataframe = pd.read_csv(file, sep=',', dtype={'Excluded_Points': str, 'Compound QC': str, 'smiles': str})
print(dataframe)
print(dataframe.columns)
