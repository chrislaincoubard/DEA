### UMAP ###
import csv
import itertools
import time
from enum import Enum

import numpy as np
import umap.umap_ as umap
from scipy.spatial import distance
from sklearn.datasets import make_blobs
from scipy import stats
from numpy.random import uniform
from scipy.stats import shapiro
from sklearn.neighbors import NearestNeighbors

np.random.seed(10)

"""
- Les 3 classes énumérations correspondent aux différentes valeurs de 
paramètre utilisé pour le benchmark, chaque paramètre testé correspond 
a une classe énumération.

- La liste des différents type de distance utilisé comme métrique.
"""


class NNeighbors(Enum):
    first = 2
    second_default = 15
    third = 40


class MinDist(Enum):
    first_default = 0.1
    second = 0.5
    third = 0.99


class NComponents(Enum):
    first = 1
    second_default = 2
    third = 3


"""
- Utilise la variable globale X_data dans laquelle les données généré sont stocké
puis execute l'algorithme Umap sur cette variable.

- Retourne les données traitée par l'agorithme Umap.

- Prend en paramètre n_neighbors, min_dist, n_components et metric, les paramètres
de l'aglorithme Umap.
"""


def use_umap(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean'):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    u = fit.fit_transform(X_data)

    return u


"""
- Calcule la distance euclidienne entre les points des données avant et après l'utilisation 
d'Umap puis calcul la déformation de cette distance entre ces deux jeux de données.

- Retourne une liste des listes des distances avant et après traitement par Umap et la déformation 
totale calculée.

- Prend en paramètre les données traitée par l'algorithme Umap (UMAP_data) et les données non traité 
par l'algorithme (real_data).
"""


def calcul_deformation(UMAP_data, real_data):
    pre_data = []

    post_data = []

    deformation = 0

    for i, j in list(itertools.combinations(real_data, 2)):
        dist_reel = distance.euclidean(i, j)
        pre_data.append(dist_reel)

    for i, j in list(itertools.combinations(UMAP_data, 2)):
        dist_UMAP = distance.euclidean(i, j)
        post_data.append(dist_UMAP)

    for i in range(len(pre_data)):
        deformation += 1 / pow(pre_data[i], 2) * pow(pre_data[i] - post_data[i], 2)
    return deformation, pre_data, post_data


"""
- Sépare les distances euclidienne calculée en 2 catégories les petites distances et les grande distances
en fonction de la valeur de cette distance par rapport a la médiane et toutes les distances.

- Retourne une liste des 4 listes : la liste des grandes et la liste des petites distance avant traitement 
par Umap et la liste des grandes et petites distance après traitement par Umap.

- Prend en paramètre la liste des distance avant et après traitement par Umap.
"""


def separate_distance(dist_reel, dist_UMAP):
    big_dist_reel, small_dist_reel = [], []
    big_dist_umap, small_dist_umap = [], []
    for index, value in enumerate(dist_reel):
        if value > np.median(dist_reel):
            big_dist_reel.append(value)
            big_dist_umap.append(dist_UMAP[index])
        else:
            small_dist_reel.append(value)
            small_dist_umap.append(dist_UMAP[index])
    return big_dist_reel, big_dist_umap, small_dist_reel, small_dist_umap


"""
- Calcule le ratio entre des données avant et après traitement par Umap, détermine le minimum et le maximum, 
les utilises pour crée une loi uniforme, effectue un test de Kolmogorov-Smirnov(K-test) entre la liste des ratio calculé 
et la loi uniforme crée, effectue un test de shapiro pour tester la normalité de la liste des ratios et calcul la 
moyenne et l'écart-type de cette liste.

- Retourne sous forme de liste le résultats du K-test, l'écart-type, la moyenne puis la résultat du test de Shapiro.

- Prend en paramètre des données avant et après traitement par Umap
"""


def ratio(data_reel, data_UMAP):
    liste_ratio = []
    for reel, umap_data in zip(data_reel, data_UMAP):
        try:
            liste_ratio.append(reel / umap_data)
        except ZeroDivisionError:
            continue
    mini = min(liste_ratio)
    maxi = max(liste_ratio)
    uni = uniform(mini, maxi, 1000)
    uni_test = stats.kstest(liste_ratio, uni)
    norm_test = shapiro(liste_ratio)
    mean = sum(liste_ratio) / len(liste_ratio)
    ecart_type = np.std(liste_ratio)
    return uni_test[1], ecart_type, mean, norm_test[1]


"""
- Calcule l'indice de Jaccard en utilisant l'algorithme de machine learning K-nearest neighbors pour trouver les plus 
proches voisins de chaque points

- Retourne la somme des indices de jaccard calculé pour chaque point.

- Prend en paramètre umap_data la liste des données après traitement par Umap et n_nbrs le nombre de voisin utilisé.
"""


def calcul_jaccard_similarity(umap_data, n_nbrs):
    neighbors_umap_data = NearestNeighbors(n_neighbors=n_nbrs, algorithm='ball_tree').fit(umap_data)
    distances_umap_data, indices_umap_data = neighbors_umap_data.kneighbors(umap_data)
    neighbors_true_data = NearestNeighbors(n_neighbors=n_nbrs, algorithm='ball_tree').fit(X_data)
    distances_true_data, indices_true_data = neighbors_true_data.kneighbors(X_data)

    sum_jaccard = 0
    for point in range(len(X_data)):
        try:
            sum_jaccard += (len(set(indices_true_data[point]) & set(indices_umap_data[point])) / len(
                set(indices_true_data[point]) | set(indices_umap_data[point])))
        except ZeroDivisionError:
            continue
    return sum_jaccard


"""
- Le main du benchmark, il ouvre ou crée un fichier umap_benchmark_ratio.csv afin de le remplir avec les 
résultats du benchmark. Constitué de 3 boucle principale dédié a chaque paramètre testé (nneighbors, 
minimum distance, nombre de composant) il appliquera les fonctions implémenté précédement afin d'obtenir 
les résultats statistique voulut.
"""
nb_columns_test = [5, 15, 25, 50, 100]
nb_rows_test = [100, 500, 1000, 5000]

with open('umap_benchmark_ratio.csv', mode='w') as umap_benchmark:
    umap_benchmark_writer = csv.writer(umap_benchmark, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    umap_benchmark_writer.writerow(
        ['nb_lines', 'nb_rows', 'n_neighbors', 'min_dist', 'n_components', 'deformation', 'cpu_time',
         "p-value uni big", "p-value norm big", "mean ratio big", "std ratio big",
         "p-value uni small", "p-value norm small", "mean ratio small", "std ratio small", "jaccard similarity"
         ])
    for nb_row in nb_rows_test:
        for nb_column in nb_columns_test:
            #### Génération de données ####

            X_data, Y_data = make_blobs(n_samples=nb_row, n_features=nb_column, centers=3, shuffle=True,
                                        random_state=10)
            #### umap ####
            for param_value in NNeighbors:
                start_cpu_time = time.process_time()
                umap_data = use_umap(param_value.value, MinDist.first_default.value, NComponents.second_default.value)
                end_cpu_time = time.process_time()
                deformation = calcul_deformation(umap_data, X_data)
                jaccard = calcul_jaccard_similarity(umap_data, param_value.value)
                sep_dist = separate_distance(deformation[1], deformation[2])
                ratio_big = ratio(sep_dist[0], sep_dist[1])
                ratio_small = ratio(sep_dist[2], sep_dist[3])
                prog_cpu_time = end_cpu_time - start_cpu_time
                umap_benchmark_writer.writerow(
                    [nb_row, nb_column, param_value.value, MinDist.first_default.value,
                     NComponents.second_default.value,
                     mean_reel, mean_UMAP, std_reel, std_UMAP,
                     deformation[0],
                     prog_cpu_time,
                     ratio_big[0], ratio_big[3], ratio_big[1], ratio_big[2],
                     ratio_small[0], ratio_small[3], ratio_small[1], ratio_small[2]], jaccard)
                print("Test du paramètre Nneighbors terminé")

            for param_value in MinDist:
                start_cpu_time = time.process_time()
                umap_data = use_umap(NNeighbors.second_default.value, param_value.value,
                                      NComponents.second_default.value)
                end_cpu_time = time.process_time()
                deformation = calcul_deformation(umap_data, X_data)
                jaccard = calcul_jaccard_similarity(umap_data, NNeighbors.second_default.value)
                sep_dist = separate_distance(deformation[1], deformation[2])
                ratio_big = ratio(sep_dist[0], sep_dist[1])
                ratio_small = ratio(sep_dist[2], sep_dist[3])
                prog_cpu_time = end_cpu_time - start_cpu_time
                umap_benchmark_writer.writerow(
                    [nb_row, nb_column, NNeighbors.second_default.value, param_value.value,
                     NComponents.second_default.value,
                     mean_reel, mean_UMAP, std_reel, std_UMAP,
                     deformation[0],
                     prog_cpu_time,
                     ratio_big[0], ratio_big[3], ratio_big[1], ratio_big[2],
                     ratio_small[0], ratio_small[3], ratio_small[1], ratio_small[2], jaccard
                     ])
                print("Test du paramètre MinDist terminé")

            for param_value in NComponents:
                start_cpu_time = time.process_time()
                umap_data = use_umap(NNeighbors.second_default.value, MinDist.first_default.value, param_value.value)
                end_cpu_time = time.process_time()
                deformation = calcul_deformation(umap_data, X_data)
                jaccard = calcul_jaccard_similarity(umap_data, NNeighbors.second_default.value)
                sep_dist = separate_distance(deformation[1], deformation[2])
                ratio_big = ratio(sep_dist[0], sep_dist[1])
                ratio_small = ratio(sep_dist[2], sep_dist[3])
                prog_cpu_time = end_cpu_time - start_cpu_time
                umap_benchmark_writer.writerow(
                    [nb_row, nb_column, NNeighbors.second_default.value, MinDist.first_default.value, param_value.value,
                     mean_reel, mean_UMAP, std_reel, std_UMAP,
                     deformation[0],
                     prog_cpu_time,
                     ratio_big[0], ratio_big[3], ratio_big[1], ratio_big[2],
                     ratio_small[0], ratio_small[3], ratio_small[1], ratio_small[2], jaccard
                     ])
                print("Test du paramètre Ncomponents terminé")
