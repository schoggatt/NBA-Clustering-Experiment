#!/usr/bin/env python

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

## Returns the label from a array of length two.
def sortByLabel(entry):
    return entry[1]

## Builds a list of two value arrays that then are sorted and displayed by their grouping.
def buildAndDisplayGroupings(model, full_set, features_set):
    general_list = []
    for index, row in full_set.iterrows():
        name = row['Player']
        column_row = features_set.loc[ full_set['Player'] == name,: ]

        row_list = column_row.values.tolist()
        row_label = model.predict(row_list)

        general_list.append([name, row_label[0]])

    general_list.sort(key = sortByLabel)
    prior_label = -1
    for index in range(len(general_list)):
        if prior_label != general_list[index][1]:
            print("\nGroup -- {}\n".format(general_list[index][1]))
        print(general_list[index][0])
        prior_label = general_list[index][1]

per_game_set = pd.read_csv("archive/nba_2020_per_game.csv")
per_game_set.drop_duplicates(subset = "Player", keep = False, inplace = True)
per_game_set.dropna(inplace = True)

## Printing Mean Values ##

print("\nMean Values\n")

print(per_game_set.mean())

## Model for General Classification

general_kmeans_model = KMeans(n_clusters = 8, random_state = 1)
good_columns = per_game_set._get_numeric_data().dropna()
good_columns.drop(columns = ["Age", "G", "GS", "G", "GS", "MP", "FT%"])
general_kmeans_model.fit(good_columns)
labels = general_kmeans_model.labels_

print("\nTop Tier of Players in NBA\n")

buildAndDisplayGroupings(general_kmeans_model, per_game_set, good_columns)

## Model for Shooter Classification ##

advanced_stat_set = pd.read_csv("archive/nba_2020_advanced.csv")

summative_stat_set = pd.merge(advanced_stat_set, per_game_set, on = "Player")

summative_stat_set.drop_duplicates(subset = "Player", keep = False, inplace = True)

shooting_kmeans_model = KMeans(n_clusters = 5, random_state = 1)
shooting_good_columns = summative_stat_set[['3P%', 'eFG%', '3PA', 'TS%', '3PAr']]._get_numeric_data().dropna()
shooting_kmeans_model.fit(shooting_good_columns)
labels = shooting_kmeans_model.labels_

print("\nTop Tier of Shooters in NBA")

buildAndDisplayGroupings(shooting_kmeans_model, summative_stat_set, shooting_good_columns)
