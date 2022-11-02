### UMAP ###
import csv
import itertools
import time
from enum import Enum

import numpy as np
import seaborn as sns
import umap.umap_ as umap
from scipy.spatial import distance
from sklearn.datasets import make_blobs
from scipy import stats
from numpy.random import uniform
from scipy.stats import shapiro
from sklearn.neighbors import NearestNeighbors

np.random.seed(10)


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


metrics = ["euclidean", "manhattan", "mahalanobis", "correlation"]


def draw_umap(data, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', title=''):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    u = fit.fit_transform(data)

    # plt.scatter(u[:, 0], u[:, 1], c=Y_data)
    # plt.title('UMAP embedding of random colours')
    # plt.show()
    return u


sns.set(style='white', context='poster', rc={'figure.figsize': (14, 10)})


#### Création des listes de distance pré et post déformation ####

def calcul_distances(data) :
    distances = []
    for i, j in list(itertools.combinations(data, 2)):
        dist = distance.euclidean(i,j)
        distances.append(dist)
    return distances

def calcul_deformation(UMAP_data, real_data):
    deformation = 0
    for i in range(len(real_data)):
        deformation += 1 / pow(real_data[i], 2) * pow(real_data[i] - UMAP_data[i], 2)
    return deformation


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

def calcul_jaccard_similarity(umap_data, n_nbrs):
    neighbors_umap_data = NearestNeighbors(n_neighbors=n_nbrs, algorithm='ball_tree').fit(umap_data)
    distances_umap_data, indices_umap_data = neighbors_umap_data.kneighbors(umap_data)
    neighbors_true_data = NearestNeighbors(n_neighbors=n_nbrs, algorithm='ball_tree').fit(X_data)
    distances_true_data, indices_true_data = neighbors_true_data.kneighbors(X_data)

    jaccard = 0
    for point in range(len(X_data)):
        try:
            jaccard += (len(set(indices_true_data[point]) & set(indices_umap_data[point])) / len(
                set(indices_true_data[point]) | set(indices_umap_data[point])))
        except ZeroDivisionError:
            continue
    return jaccard

#### Lancement Umap sans graphique ####
# params = [NNeighbors, MinDist, NComponents]
nb_columns_test = [5, 15, 25, 50, 100]
nb_rows_test = [100, 500, 1000]

with open('umap_benchmark_ratio.csv', mode='w') as umap_benchmark:
    umap_benchmark_writer = csv.writer(umap_benchmark, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    umap_benchmark_writer.writerow(
        ['nb_lines', 'nb_rows', 'n_neighbors', 'min_dist', 'n_components', "moyenne reel", "moyenne UMAP", "std reel",
         "std UMAP", 'deformation', 'cpu_time',
         "p-value uniformité global", "p-value normalité global", "std ratio global", " mean ratio global",
         "p-value uni big", "p-value norm big", "mean ratio big", "std ratio big",
         "p-value uni small", "p-value norm small", "mean ratio small", "std ratio small",
         "jaccard similarity"
         ])
    for nb_row in nb_rows_test:
        for nb_column in nb_columns_test:
            #### Génération de données ####

            X_data, Y_data = make_blobs(n_samples=nb_row, n_features=nb_column, centers=3, shuffle=True,
                                        random_state=10)
            #### umap ####
            dist_reel = calcul_distances(X_data)
            for param_value in NNeighbors:
                print(f"Start for this one : rows : {nb_row}, cols {nb_column}, param_value {param_value.value}")
                start_cpu_time = time.process_time()
                Xtrem_data = draw_umap(X_data,param_value.value, MinDist.first_default.value, NComponents.second_default.value)
                end_cpu_time = time.process_time()
                dist_UMAP = calcul_distances(Xtrem_data)
                deformation = calcul_deformation(dist_UMAP, dist_reel)
                jaccard = calcul_jaccard_similarity(Xtrem_data, param_value.value)
                mean_reel = sum(dist_reel) / len(dist_reel)
                mean_UMAP = sum(dist_UMAP) / len(dist_UMAP)
                sep_dist = separate_distance(dist_reel, dist_UMAP)
                std_reel = np.std(np.array(dist_reel))
                std_UMAP = np.std(np.array(dist_UMAP))
                ratio_all = ratio(dist_reel, dist_UMAP)
                ratio_big = ratio(sep_dist[0], sep_dist[1])
                ratio_small = ratio(sep_dist[2], sep_dist[3])
                prog_cpu_time = end_cpu_time - start_cpu_time
                umap_benchmark_writer.writerow(
                    [nb_row, nb_column, param_value.value, MinDist.first_default.value,
                     NComponents.second_default.value,
                     mean_reel, mean_UMAP, std_reel, std_UMAP,
                     deformation,
                     prog_cpu_time,
                     ratio_all[0], ratio_all[3], ratio_all[1], ratio_all[2],
                     ratio_big[0], ratio_big[3], ratio_big[1], ratio_big[2],
                     ratio_small[0], ratio_small[3], ratio_small[1], ratio_small[2],
                     jaccard])
                print(f"Done for this one : rows : {nb_row}, cols {nb_column}, param_value {param_value.value}")

            for param_value in MinDist:
                print(f"Start for this one : rows : {nb_row}, cols {nb_column}, param_value {param_value.value}")
                start_cpu_time = time.process_time()
                Xtrem_data = draw_umap(X_data,NNeighbors.second_default.value, param_value.value, NComponents.second_default.value)
                end_cpu_time = time.process_time()
                dist_UMAP = calcul_distances(Xtrem_data)
                deformation = calcul_deformation(dist_UMAP, dist_reel)
                jaccard = calcul_jaccard_similarity(Xtrem_data, NNeighbors.second_default.value)
                mean_reel = sum(dist_reel) / len(dist_reel)
                mean_UMAP = sum(dist_UMAP) / len(dist_UMAP)
                sep_dist = separate_distance(dist_reel, dist_UMAP)
                std_reel = np.std(np.array(dist_reel))
                std_UMAP = np.std(np.array(dist_UMAP))
                ratio_all = ratio(dist_reel, dist_UMAP)
                ratio_big = ratio(sep_dist[0], sep_dist[1])
                ratio_small = ratio(sep_dist[2], sep_dist[3])
                prog_cpu_time = end_cpu_time - start_cpu_time
                umap_benchmark_writer.writerow(
                    [nb_row, nb_column, param_value.value, MinDist.first_default.value,
                     NComponents.second_default.value,
                     mean_reel, mean_UMAP, std_reel, std_UMAP,
                     deformation,
                     prog_cpu_time,
                     ratio_all[0], ratio_all[3], ratio_all[1], ratio_all[2],
                     ratio_big[0], ratio_big[3], ratio_big[1], ratio_big[2],
                     ratio_small[0], ratio_small[3], ratio_small[1], ratio_small[2],
                     jaccard])
                print(f"Done for this one : rows : {nb_row}, cols {nb_column}, param_value {param_value.value}")

            for param_value in NComponents:
                print(f"Start for this one : rows : {nb_row}, cols {nb_column}, param_value {param_value.value}")
                start_cpu_time = time.process_time()
                Xtrem_data = draw_umap(X_data,NNeighbors.second_default.value, MinDist.first_default.value, param_value.value)
                end_cpu_time = time.process_time()
                dist_UMAP = calcul_distances(Xtrem_data)
                deformation = calcul_deformation(dist_UMAP, dist_reel)
                jaccard = calcul_jaccard_similarity(Xtrem_data, NNeighbors.second_default.value)
                mean_reel = sum(dist_reel) / len(dist_reel)
                mean_UMAP = sum(dist_UMAP) / len(dist_UMAP)
                sep_dist = separate_distance(dist_reel, dist_UMAP)
                std_reel = np.std(np.array(dist_reel))
                std_UMAP = np.std(np.array(dist_UMAP))
                ratio_all = ratio(dist_reel, dist_UMAP)
                ratio_big = ratio(sep_dist[0], sep_dist[1])
                ratio_small = ratio(sep_dist[2], sep_dist[3])
                prog_cpu_time = end_cpu_time - start_cpu_time
                umap_benchmark_writer.writerow(
                    [nb_row, nb_column, param_value.value, MinDist.first_default.value,
                     NComponents.second_default.value,
                     mean_reel, mean_UMAP, std_reel, std_UMAP,
                     deformation,
                     prog_cpu_time,
                     ratio_all[0], ratio_all[3], ratio_all[1], ratio_all[2],
                     ratio_big[0], ratio_big[3], ratio_big[1], ratio_big[2],
                     ratio_small[0], ratio_small[3], ratio_small[1], ratio_small[2],
                     jaccard])
                print(f"Done for this one : rows : {nb_row}, cols {nb_column}, param_value {param_value.value}")
