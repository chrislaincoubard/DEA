### UMAP ###
from multiprocessing import context
import seaborn as sns
import umap
from sklearn.datasets import make_blobs
from scipy.spatial import distance


def draw_umap(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', title=''):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    u = fit.fit_transform(X_data)

    # plt.scatter(u[:,0], u[:,1], c=Y_data)
    # plt.title('UMAP embedding of random colours')
    # plt.show()
    return u

sns.set(style='white', context='poster', rc={'figure.figsize':(14,10)})

#### Génération de données ####

X_data, Y_data = make_blobs(n_samples=100, n_features=8, centers=3,shuffle=True)

#### Lancement Umap sans graphique ####

Xtrem_data = draw_umap(4,0.5,2)

#### Création des listes de distance pré et post déformation ####

pre_data = []

post_data = []

deformation = 0

for i in range(len(X_data)):
    for j in range (len(X_data)):
        if i < j :
            pre_data.append(distance.euclidean(X_data[i],X_data[j]))

for i in range(len(Xtrem_data)):
    for j in range (len(Xtrem_data)):
        if i < j :
            post_data.append(distance.euclidean(Xtrem_data[i],Xtrem_data[j]))

for i in range(len(pre_data)):
    deformation +=  1/pow(pre_data[i],2) * pow(pre_data[i] - post_data[i],2)

#### RESULTATS #### WORK IN PROGRESS
print("proto-déformation : ",deformation) # Apriori ca fonctionne mais je suis sur de rien pour l'instant
