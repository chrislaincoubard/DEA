import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px

df = pd.read_csv('umap_stats.csv')
Y = []  # liste des labels

i = 0
for index in df.iterrows():
    Y.append(i) # Y ci-dessous permet d'avoir les couleurs en gradient
    i += 1

X = df.drop(["nb_lines", "nb_rows", "n_neighbors", "min_dist", "n_components"],axis=1)

X_clean = df.drop([36, 135]) #- Suppression [50, 100, 2, 0.1, 2] et [1000, 5, 2, 0.1, 2] car elles écrasaient le graphe
Y_clean = []
i = 0
for index in X_clean.iterrows():
    Y_clean.append(i)
    i += 1

"""
- Graphique de l'analyse en composante principale sur toute les conditions pour la similarité de Jaccard, le temps de 
calcul (cpu_time) et la déformation des distances 
"""
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

"""
- Graphique de l'analyse en composante principale sur toutes les conditions pour la similarité de jaccad, le temps de 
calcul, la valeur de déformation et le coefficient de variation du ratio.
"""
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

"""
- Génération des histrogrammes.
"""

X_clean = df.drop([90, 135]) #- Supression [500	25	2	0,1	2] et [1000, 5, 2, 0.1, 2] car elles écrasaient le graphe

"""
- Calcul du coefficient de variation et récupération des autres mesures.
"""
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

"""
Graphique présentant le coefficient de variation du ratio entre les grandes distances réelles et leur
équivalent entre les mêmes points après UMAP pour toutes les conditions expérimentales.
"""
fig_CV_big = px.bar(CV_ratio_big,
                    labels={"value": "Coefficient Variation Grandes Distances (%)", "index": "Conditions"})
fig_CV_big.update_layout(showlegend=False)
fig_CV_big.show()

"""
- Graphique présentant le coefficient de variation du ratio entre les petites distances réelles et leur
équivalent entre les mêmes points après UMAP pour toutes les conditions expérimentales.
"""
fig_CV_small = px.bar(CV_ratio_small,
                      labels={"value": "Coefficient Variation Petites Distances (%)", "index": "Conditions"})
fig_CV_small.update_layout(showlegend=False)
fig_CV_small.show()
"""
- Graphique valeur de déformation sans le outlier infini.
"""
fig_deformation = px.bar(deformation, labels={"value": "Deformation", "index": "Conditions"})
fig_deformation.update_layout(showlegend=False)
fig_deformation.show()

"""
- Graphique présentant la valeur de déformation pour toutes les conditions.
"""
fig_deformation_clean = px.bar(deformation_clean, labels={"value": "Deformation", "index": "Conditions"})
fig_deformation_clean.update_layout(showlegend=False)
fig_deformation_clean.show()

"""
- Graphique présentant le temps de calcul en secondes pour toutes les conditions pour calculer la
réduction de dimension par UMAP.
"""
fig_CV_cpu = px.bar(cpu_time, labels={"value": "Temps de calcul", "index": "Conditions"})
fig_CV_cpu.update_layout(showlegend=False)
fig_CV_cpu.show()

"""
- Graphique présentant la somme des indices de Jaccard pour chaque point, pour toutes les conditions
expérimentales.
"""
fig_CV_jaccard = px.bar(jaccard, labels={"value": "Somme Coefficient Similarité Jaccard", "index": "Conditions"})
fig_CV_jaccard.update_layout(showlegend=False)
fig_CV_jaccard.show()
