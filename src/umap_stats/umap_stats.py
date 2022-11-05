import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('umap_stats.csv')
Y = []  # liste des labels
for index, row in df.iterrows():
    Y.append(str(list(row[0:5])))

X = df.drop(["nb_lines", "nb_rows", "n_neighbors", "min_dist", "n_components"],
            axis=1)  # suppr les labels pour avoir que les data



#Essai avec toutes les données
x = X.iloc[:, :].values
print(x)
x = StandardScaler().fit_transform(x)

pca_df = PCA(n_components=2)
principalComponents_df = pca_df.fit_transform(x)
principal_Df = pd.DataFrame(data=principalComponents_df
                            , columns=['principal component 1', 'principal component 2'])
print('Explained variation per principal component: {}'.format(pca_df.explained_variance_ratio_))

fig = px.scatter(principal_Df, x="principal component 1", y="principal component 2", color=Y,
                 labels={'color': 'Sous type'})
fig.show()

# 3D

pca_3Ddf = PCA(n_components=3)
principalComponents_3Ddf = pca_3Ddf.fit_transform(x)
principal_3DDf = pd.DataFrame(data=principalComponents_3Ddf
                              , columns=['principal component 1', 'principal component 2', 'principal component 3'])
print('Explained variation per principal component 3D: {}'.format(pca_3Ddf.explained_variance_ratio_))

fig_3D = px.scatter_3d(principal_3DDf, x="principal component 1", y="principal component 2", z="principal component 3",
                       color=Y, labels={'color': 'Sous type'})
fig_3D.show()

##########################################################
#essai que des 6 premières (faut rajouter jaccard je pense)
x = X.iloc[:, :6].values
print(x)
x = StandardScaler().fit_transform(x)

pca_df = PCA(n_components=2)
principalComponents_df = pca_df.fit_transform(x)
principal_Df = pd.DataFrame(data=principalComponents_df
                            , columns=['principal component 1', 'principal component 2'])
print('Explained variation per principal component: {}'.format(pca_df.explained_variance_ratio_))

fig = px.scatter(principal_Df, x="principal component 1", y="principal component 2", color=Y,
                 labels={'color': 'Sous type'})
fig.show()

# 3D

pca_3Ddf = PCA(n_components=3)
principalComponents_3Ddf = pca_3Ddf.fit_transform(x)
principal_3DDf = pd.DataFrame(data=principalComponents_3Ddf
                              , columns=['principal component 1', 'principal component 2', 'principal component 3'])
print('Explained variation per principal component 3D: {}'.format(pca_3Ddf.explained_variance_ratio_))

fig_3D = px.scatter_3d(principal_3DDf, x="principal component 1", y="principal component 2", z="principal component 3",
                       color=Y, labels={'color': 'Sous type'})
fig_3D.show()

#################################################################
#faire ici un autre test (avec les colonnes des p-values et ti=out ça donne rien
x = X.iloc[:, 6:].values
print(x)
x = StandardScaler().fit_transform(x)

pca_df = PCA(n_components=2)
principalComponents_df = pca_df.fit_transform(x)
principal_Df = pd.DataFrame(data=principalComponents_df
                            , columns=['principal component 1', 'principal component 2'])
print('Explained variation per principal component: {}'.format(pca_df.explained_variance_ratio_))

fig = px.scatter(principal_Df, x="principal component 1", y="principal component 2", color=Y,
                 labels={'color': 'Sous type'})
fig.show()

# 3D

pca_3Ddf = PCA(n_components=3)
principalComponents_3Ddf = pca_3Ddf.fit_transform(x)
principal_3DDf = pd.DataFrame(data=principalComponents_3Ddf
                              , columns=['principal component 1', 'principal component 2', 'principal component 3'])
print('Explained variation per principal component 3D: {}'.format(pca_3Ddf.explained_variance_ratio_))

fig_3D = px.scatter_3d(principal_3DDf, x="principal component 1", y="principal component 2", z="principal component 3",
                       color=Y, labels={'color': 'Sous type'})
fig_3D.show()


########################################################################
#Calcul du coefficient de variation

CV = []
CV_ratio_big = []
CV_ratio_small = []


for index, row in X.iterrows(): #iterate over rows
        CV.append((row["moyenne reel"] / row["std reel"]) * 100)
        CV_ratio_big.append((row["mean ratio big"] / row["std ratio big"]) * 100)
        CV_ratio_small.append((row["mean ratio small"] / row["std ratio small"]) * 100)


corresp_index_conditions = {}
for index, value in enumerate(CV_ratio_big):
    corresp_index_conditions[value] = index

fig_CV_big = plt.bar(list(corresp_index_conditions.values()), CV_ratio_big)

plt.show()

