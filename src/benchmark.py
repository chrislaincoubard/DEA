### UMAP ###
import time
import seaborn as sns
import numpy as np
import umap
from sklearn.datasets import make_blobs
from scipy.spatial import distance
import matplotlib.pyplot as plt
import csv
from enum import Enum

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

    for i in range(len(X_data)):
        for j in range(len(X_data)):
            if i < j:
                pre_data.append(distance.euclidean(X_data[i], X_data[j]))

    for i in range(len(Xtrem_data)):
        for j in range(len(Xtrem_data)):
            if i < j:
                post_data.append(distance.euclidean(Xtrem_data[i], Xtrem_data[j]))

    for i in range(len(pre_data)):
        deformation += 1 / pow(pre_data[i], 2) * pow(pre_data[i] - post_data[i], 2)
    return deformation


#### Lancement Umap sans graphique ####
# params = [NNeighbors, MinDist, NComponents]
nb_rows_test = [5, 15, 25]
nb_lines_test = [100, 500, 1000]

with open('umap_benchmark.csv', mode='w') as umap_benchmark:
    umap_benchmark_writer = csv.writer(umap_benchmark, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    umap_benchmark_writer.writerow(
        ['nb_lines', 'nb_rows', 'n_neighbors', 'min_dist', 'n_components', 'deformation', 'cpu_time'])
    for nb_line in nb_lines_test:
        for nb_row in nb_rows_test:
            #### Génération de données ####

            X_data, Y_data = make_blobs(n_samples=nb_line, n_features=nb_row, centers=3, shuffle=True)

            #### umap ####
            for param_value in NNeighbors:
                start_cpu_time = time.process_time()
                Xtrem_data = draw_umap(param_value.value, MinDist.first_default.value, NComponents.second_default.value)
                deformation = calcul_deformation(Xtrem_data)
                end_cpu_time = time.process_time()
                prog_cpu_time = end_cpu_time - start_cpu_time
                umap_benchmark_writer.writerow(
                    [nb_line, nb_row, param_value.value, MinDist.first_default.value, NComponents.second_default.value,
                     deformation,
                     prog_cpu_time])

            for param_value in MinDist:
                start_cpu_time = time.process_time()
                Xtrem_data = draw_umap(NNeighbors.second_default.value, param_value.value,
                                       NComponents.second_default.value)
                deformation = calcul_deformation(Xtrem_data)
                end_cpu_time = time.process_time()
                prog_cpu_time = end_cpu_time - start_cpu_time
                umap_benchmark_writer.writerow(
                    [nb_line, nb_row, NNeighbors.second_default.value, param_value.value,
                     NComponents.second_default.value, deformation,
                     prog_cpu_time])

            for param_value in NComponents:
                start_cpu_time = time.process_time()
                Xtrem_data = draw_umap(NNeighbors.second_default.value, MinDist.first_default.value, param_value.value)
                deformation = calcul_deformation(Xtrem_data)
                end_cpu_time = time.process_time()
                prog_cpu_time = end_cpu_time - start_cpu_time
                umap_benchmark_writer.writerow(
                    [nb_line, nb_row, NNeighbors.second_default.value, MinDist.first_default.value, param_value.value,
                     deformation,
                     prog_cpu_time])
