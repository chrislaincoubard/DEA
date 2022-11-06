import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px

df = pd.read_csv('umap_stats.csv')
Y = []  # liste des labels

# Y ci-dessous permet d'avoir les couleurs en gradient
i = 0
for index in df.iterrows():
    Y.append(i)
    i += 1

X = df.drop(["nb_lines", "nb_rows", "n_neighbors", "min_dist", "n_components"],
            axis=1)  # suppr les labels pour avoir que les data

# avec les conditions [50, 100, 2, 0.1, 2] et [1000, 5, 2, 0.1, 2] en moins car elles écrasaient le graphe
X_clean = df.drop([36, 135])
Y_clean = []
i = 0
for index in X_clean.iterrows():
    Y_clean.append(i)
    i += 1

##########################################################
# PCA deformation, cpu_time, jaccard avec les points déformants
x = X.iloc[:, [4, 5, 18]].values

x = StandardScaler().fit_transform(x)

pca_df = PCA(n_components=2)
principalComponents_df = pca_df.fit_transform(x)
principal_Df = pd.DataFrame(data=principalComponents_df
                            , columns=['principal component 1', 'principal component 2'])
print('Explained variation per principal component: {}'.format(pca_df.explained_variance_ratio_))

fig = px.scatter(principal_Df, x="principal component 1", y="principal component 2", color=Y,
                 labels={'color': 'Conditions'})
fig.show()

##########################################################
# PCA deformation, cpu_time, jaccard, std ratio small sans point déformant
x = X_clean.iloc[:, [4, 5, 17, 18]].values
x = StandardScaler().fit_transform(x)

pca_df = PCA(n_components=2)
principalComponents_df = pca_df.fit_transform(x)
principal_Df = pd.DataFrame(data=principalComponents_df
                            , columns=['principal component 1', 'principal component 2'])
print('Explained variation per principal component: {}'.format(pca_df.explained_variance_ratio_))

fig = px.scatter(principal_Df, x="principal component 1", y="principal component 2", color=Y_clean,
                 labels={'color': 'Conditions'})
fig.show()

########################################################################
# Génération des graphes
# avec les conditions [500	25	2	0,1	2] et [1000, 5, 2, 0.1, 2] en moins car elles écrasaient le graphe
X_clean = df.drop([90, 135])
# Calcul du coefficient de variation et récupération des autres mesures
CV_ratio_big = []
CV_ratio_small = []
deformation = []
deformation_clean = []
cpu_time = []
jaccard = []
nb_rows = []

for index, row in X.iterrows():  # iterate over rows
    CV_ratio_big.append((row["std ratio big"] / row["mean ratio big"]) * 100)
    CV_ratio_small.append((row["std ratio small"] / row["mean ratio small"]) * 100)
    cpu_time.append(row["cpu_time"])
    deformation.append(row["deformation"])
    jaccard.append(row["jaccard similarity"])


for index, row in X_clean.iterrows():  # pour la déformation sans le point en trop
    deformation_clean.append(row["deformation"])

# ratio big distances
fig_CV_big = px.bar(CV_ratio_big,
                    labels={"value": "Coefficient Variation Grandes Distances (%)", "index": "Conditions"})
fig_CV_big.update_layout(showlegend=False)
fig_CV_big.show()

# ratio small distances
fig_CV_small = px.bar(CV_ratio_small,
                      labels={"value": "Coefficient Variation Petites Distances (%)", "index": "Conditions"})
fig_CV_small.update_layout(showlegend=False)
fig_CV_small.show()

# deformation sans clean
fig_deformation = px.bar(deformation, labels={"value": "Deformation", "index": "Conditions"})
fig_deformation.update_layout(showlegend=False)
fig_deformation.show()
# deformation avec clean
fig_deformation_clean = px.bar(deformation_clean, labels={"value": "Deformation", "index": "Conditions"})
fig_deformation_clean.update_layout(showlegend=False)
fig_deformation_clean.show()

# ratio cpu_time
fig_CV_cpu = px.bar(cpu_time, labels={"value": "Temps de calcul", "index": "Conditions"})
fig_CV_cpu.update_layout(showlegend=False)
fig_CV_cpu.show()

# ratio jaccard
fig_CV_jaccard = px.bar(jaccard, labels={"value": "Somme Coefficient Similarité Jaccard", "index": "Conditions"})
fig_CV_jaccard.update_layout(showlegend=False)
fig_CV_jaccard.show()
