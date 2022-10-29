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
def draw_umap(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', title=''):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    u = fit.fit_transform(X_data)

    # plt.scatter(u[:, 0], u[:, 1], c=Y_data)
    # plt.title('UMAP embedding of random colours')
    # plt.show()
    return u


sns.set(style='white', context='poster', rc={'figure.figsize': (14, 10)})


#### Création des listes de distance pré et post déformation ####
def calcul_deformation(Xtrem_data):
    pre_data = []

    post_data = []

    deformation = 0

    for i, j in list(itertools.combinations(X_data, 2)):
        dist_reel = distance.euclidean(i, j)
        pre_data.append(dist_reel)

    for i, j in list(itertools.combinations(Xtrem_data, 2)):
        dist_UMAP = distance.euclidean(i, j)
        post_data.append(dist_UMAP)

    for i in range(len(pre_data)):
        deformation += 1 / pow(pre_data[i], 2) * pow(pre_data[i] - post_data[i], 2)
    print(pre_data)
    print(post_data)
    return deformation, pre_data, post_data

def ratio(data_reel, data_UMAP):
    liste_ratio = []
    for reel, umap_data in zip(data_reel, data_UMAP):
        try :
            liste_ratio.append(reel / umap_data)
        except ZeroDivisionError:
            continue
    uni_test = stats.kstest(liste_ratio, stats.norm.cdf)
    mean = sum(liste_ratio) / len(liste_ratio)
    ecart_type = np.std(liste_ratio)
    return uni_test, ecart_type, mean

#### Lancement Umap sans graphique ####
# params = [NNeighbors, MinDist, NComponents]
nb_columns_test = [5, 15, 25, 50, 100]
nb_rows_test = [100, 500, 1000, 5000]

with open('umap_benchmark_ratio.csv', mode='w') as umap_benchmark:
    umap_benchmark_writer = csv.writer(umap_benchmark, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    umap_benchmark_writer.writerow(
        ['nb_lines', 'nb_rows', 'n_neighbors', 'min_dist', 'n_components', 'deformation', 'cpu_time', "Uni ratio", "std ratio"," mean ratio"])
    for nb_row in nb_rows_test:
        for nb_column in nb_columns_test:
            #### Génération de données ####

            X_data, Y_data = make_blobs(n_samples=nb_row, n_features=nb_column, centers=3, shuffle=True)
            #### umap ####
            for param_value in NNeighbors:
                start_cpu_time = time.process_time()
                Xtrem_data = draw_umap(param_value.value, MinDist.first_default.value, NComponents.second_default.value)
                end_cpu_time = time.process_time()
                deformation = calcul_deformation(Xtrem_data)
                ratio_stats = ratio(deformation[1], deformation[2])
                prog_cpu_time = end_cpu_time - start_cpu_time
                umap_benchmark_writer.writerow(
                    [nb_row, nb_column, param_value.value, MinDist.first_default.value, NComponents.second_default.value,
                     deformation,
                     prog_cpu_time,
                     ratio_stats[0], ratio_stats[1], ratio_stats[2]])

            for param_value in MinDist:
                start_cpu_time = time.process_time()
                Xtrem_data = draw_umap(NNeighbors.second_default.value, param_value.value,
                                       NComponents.second_default.value)
                end_cpu_time = time.process_time()
                deformation = calcul_deformation(Xtrem_data)
                ratio_stats = ratio(deformation[1], deformation[2])
                prog_cpu_time = end_cpu_time - start_cpu_time
                umap_benchmark_writer.writerow(
                    [nb_row, nb_column, NNeighbors.second_default.value, param_value.value,
                     NComponents.second_default.value, deformation,
                     prog_cpu_time,
                     ratio_stats[0], ratio_stats[1], ratio_stats[2]])

            for param_value in NComponents:
                start_cpu_time = time.process_time()
                Xtrem_data = draw_umap(NNeighbors.second_default.value, MinDist.first_default.value, param_value.value)
                end_cpu_time = time.process_time()
                deformation = calcul_deformation(Xtrem_data)
                ratio_stats = ratio(deformation[1], deformation[2])
                prog_cpu_time = end_cpu_time - start_cpu_time
                umap_benchmark_writer.writerow(
                    [nb_row, nb_column, NNeighbors.second_default.value, MinDist.first_default.value, param_value.value,
                     deformation,
                     prog_cpu_time,
                     ratio_stats[0], ratio_stats[1], ratio_stats[2]])
