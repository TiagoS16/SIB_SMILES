import sys
import copy
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem, DataStructs
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
from loaders.Loaders import CSVLoader
from standardizer.CustomStandardizer import CustomStandardizer
from scalers.sklearnScalers import StandardScaler
from compoundFeaturization.rdkitDescriptors import TwoDimensionDescriptors
from compoundFeaturization.rdkitFingerprints import MorganFingerprint, RDKFingerprint, MACCSkeysFingerprint
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from boruta.boruta_py import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

sys.path.append('src')
#%% md
## Initial exploration
#%% md
### Import dataset
#%%
file = '../dataset/TDP1_activity_dataset.csv'
dataset = pd.read_csv(file, sep=',')
dataset.head()
#%% md
### Simple Analyses
#%%
print(dataset.size)
#%%
print(dataset.shape)

#%%
print(dataset.columns)

#%%
print(dataset.dtypes)
#%%
sub_dataset = dataset[['Potency', 'Efficacy', 'Fit_LogAC50', 'Fit_HillSlope', 'Fit_R2',
       'Fit_InfiniteActivity', 'Fit_ZeroActivity', 'Activity at 0.0000295000 uM',
       'Activity at 0.0000590000 uM', 'Activity at 0.0001503265 uM',
       'Activity at 0.0002712146 uM', 'Activity at 0.0005895491 uM',
       'Activity at 0.00117 uM', 'Activity at 0.00179 uM',
       'Activity at 0.00299 uM', 'Activity at 0.00672 uM',
       'Activity at 0.014 uM', 'Activity at 0.026 uM', 'Activity at 0.040 uM',
       'Activity at 0.074 uM', 'Activity at 0.167 uM', 'Activity at 0.363 uM',
       'Activity at 0.628 uM', 'Activity at 0.975 uM', 'Activity at 1.849 uM',
       'Activity at 4.119 uM', 'Activity at 9.037 uM', 'Activity at 15.83 uM',
       'Activity at 21.08 uM', 'Activity at 46.23 uM', 'Activity at 92.54 uM',
       'Activity at 165.6 uM']]

sub_dataset.describe()
#%% md
## Pre-Processing

### Visualization of the NA's
#%%
print(dataset.isna().sum())
print(f"TOTAL: {dataset.isna().sum().sum()}")

### Drop specific features
#%%
dataset = dataset.dropna(axis=1, how='all')
dataset.drop(['PUBCHEM_ACTIVITY_URL', 'Compound QC'], axis=1)
dataset = dataset[dataset['smiles'].notna()]

print(dataset.shape)
print(dataset.columns)

#%%
dataset_1 = pd.DataFrame.copy(dataset)
conditions = [(-150 <= dataset_1["Activity at 46.23 uM"]) & (dataset_1["Activity at 46.23 uM"] < -100),
              (-100 <= dataset_1["Activity at 46.23 uM"]) & (dataset_1["Activity at 46.23 uM"] < -50),
              (-50 <= dataset_1["Activity at 46.23 uM"]) & (dataset_1["Activity at 46.23 uM"] < 0),
              (0 <= dataset_1["Activity at 46.23 uM"]) & (dataset_1["Activity at 46.23 uM"] < 50)]
results = ["0", "1", "2", "3"]
dataset_1['Activity at 46.23 uM'] = np.select(conditions, results)

dataset_1.to_csv("../dataset/multiclass/activity_46_multiclass.csv")

## Graphic Exploration
## realizr pie chart para ver a igualdade das classes, se estão bem distribuidas ou não
class_activity = dataset_1.groupby('Activity at 46.23 uM').size()
class_labels_activity = dataset_1.groupby('Activity at 46.23 uM').size().index()

fig, (ax1) = plt.subplots(1, figsize=(15, 5))
ax1.pie(class_activity, labels=class_labels_activity, autopct='%1.1f%%', startangle=90)
ax1.set_title('Activity Classes')
dataset_1 = None

## Compound Standardization

#%%
def standardize(dataset, id_field ,mols_field,class_field):

    loader = CSVLoader(dataset,
                       id_field=id_field,
                       mols_field = mols_field,
                       labels_fields = class_field)

    dataset = loader.create_dataset()

    standardisation_params = {
        'REMOVE_ISOTOPE': True,
        'NEUTRALISE_CHARGE': True,
        'REMOVE_STEREO': False,
        'KEEP_BIGGEST': True,
        'ADD_HYDROGEN': False,
        'KEKULIZE': True,
        'NEUTRALISE_CHARGE_LATE': True}

    CustomStandardizer(params=standardisation_params).standardize(dataset)

    return dataset
#%%
dataset = standardize("../dataset/multiclass/activity_46_multiclass.csv", "PUBCHEM_CID", "smiles", "Activity at 46.23 uM")
dataset.save_to_csv("../dataset/multiclass/standardized_multiclass.csv")
#%% md
## Feature Generation

#%%
loader = CSVLoader("../dataset/multiclass/standardized_multiclass.csv",
                   mols_field='mols',
                   labels_fields='y')

dataset_des = loader.create_dataset()
dataset_finger = copy.deepcopy(dataset_des)
#%% md
### Molecular Descriptors
#%%
calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
header = list(calc.GetDescriptorNames())
print(header)
#%%
TwoDimensionDescriptors().featurize(dataset_des)
#%% md

#%% md
#### Create dataframe with feature names
#%%
descript_data = pd.DataFrame(dataset_des.X, columns=header)
print(descript_data)
#%%
print(descript_data.shape)
#%%
descript_data.describe()
#%%
descript_data["Activity at 46.23 uM"] = dataset_des.y
print(descript_data)
#%%
descript_data.to_csv('../dataset/multiclass/descriptors_multiclass.csv', index=False)
#%%
# descript_data = pd.read_csv('../dataset/descriptors_binary.csv')
#%%
# separar o dataframe por atividade
moldes_0 = descript_data[descript_data["Activity at 46.23 uM"] == "0"]
moldes_1 = descript_data[descript_data["Activity at 46.23 uM"] == "1"]
moldes_2 = descript_data[descript_data["Activity at 46.23 uM"] == "2"]
moldes_3 = descript_data[descript_data["Activity at 46.23 uM"] == "3"]
#%%
moldes_0.describe()
#%%
moldes_1.describe()
#%%
moldes_2.describe()
#%%
moldes_3.describe()
#
# #%%
# def generate_box_plot(feature, class_name, title, dataframe, orientation):
#     plt.subplots(figsize=(20, 10))
#     sns.set(font_scale=1.4)
#     plt.title(title, fontsize=25)
#     sns.boxplot(x=feature, y=class_name, orient=orientation,
#                 data=dataframe, palette="Set3")
# #%%
# generate_box_plot("ExactMolWt", "binary_class", "", descript_data, "h")
# #%%
# moldes_1["ExactMolWt"].describe()
# #%%
# moldes_0["ExactMolWt"].describe()
# #%%
# generate_box_plot("RingCount", "binary_class", "", descript_data, "h")
# #%%
# moldes_1["RingCount"].describe()
# #%%
# moldes_0["RingCount"].describe()
# #%%
# generate_box_plot("NumAromaticRings", "binary_class", "", descript_data, "h")
# #%%
# moldes_1["NumAromaticRings"].describe()
# #%%
# moldes_0["NumAromaticRings"].describe()
#
# #%%
# generate_box_plot("TPSA", "binary_class", "", descript_data, "h")
# #%%
# moldes_1["TPSA"].describe()
# #%%
# moldes_0["TPSA"].describe()
#
#### Normalize Data
#%%
StandardScaler().fit_transform(dataset_des)

### Molecular Fingerprints

#%%
# dataset_morgan = copy.deepcopy(dataset_finger)
dataset_rdk = copy.deepcopy(dataset_finger)
# dataset_macc = copy.deepcopy(dataset_finger)
#%%
# MorganFingerprint().featurize(dataset_morgan)
#%%
# print(dataset_morgan.X.shape)
#%%
RDKFingerprint().featurize(dataset_rdk)
#%%
print(dataset_rdk.X.shape)
#%%
# MACCSkeysFingerprint().featurize(dataset_macc)
#%%
# print(dataset_macc.X.shape)

## Feature Selection

### Molecular Descriptors
#%%
rf = RandomForestClassifier(n_jobs=-1)
feat_selector = BorutaPy(estimator=rf, max_iter=10, n_estimators=100)

feat_selector.fit(X=dataset_des.X, y=dataset_des.y)
X_filtered = feat_selector.transform(X=dataset_des.X)
#%%
features = []
for i in range(len(feat_selector.support_)):
    if feat_selector.support_[i] == True:
        features.append(header[i])
#%%
descript_data = pd.DataFrame(X_filtered, columns=features)
descript_data["Activity at 46.23 uM"] = dataset_des.y
print(descript_data)
#%%
print(descript_data.shape)
#%%
### Molecular Fingerprints

# dataset_morgan_fs = SelectPercentile(percentile=10).fit_transform(dataset_morgan.X, dataset_morgan.y)
# print(dataset_morgan_fs.shape)
#%%
dataset_rdk_fs = SelectPercentile(percentile=10).fit_transform(dataset_rdk.X, dataset_rdk.y)
print(dataset_rdk_fs.shape)
#%%
# dataset_macc_fs = SelectPercentile(percentile=10).fit_transform(dataset_macc.X, dataset_macc.y)
# print(dataset_macc_fs.shape)
#%%
descript_data.to_csv("../dataset/multiclass/descriptors_fs.csv", index=False)

# pd_morgan_fs = pd.DataFrame(dataset_morgan_fs)
# pd_morgan_fs['Activity at 46.23 uM'] = dataset_morgan.y
# pd_morgan_fs.to_csv("../dataset/multiclass/morgan_fs.csv", index=False)

pd_rdk_fs = pd.DataFrame(dataset_rdk_fs)
pd_rdk_fs['Activity at 46.23 uM'] = dataset_rdk.y
pd_rdk_fs.to_csv("../dataset/multiclass/rdk_fs.csv", index=False)

# pd_macc_fs = pd.DataFrame(dataset_macc_fs)
# pd_macc_fs['Activity at 46.23 uM'] = dataset_macc.y
# pd_macc_fs.to_csv("../dataset/multiclass/macc_fs.csv", index=False)
