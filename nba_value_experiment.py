import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

print("\nTop 10 Rows\n")

per_game_set = pd.read_csv("archive/nba_2020_per_game.csv")
per_game_set.drop_duplicates(subset = "Player", keep = False, inplace = True)
per_game_set.dropna(inplace = True)

print("\nMean Values\n")

print(per_game_set.mean())

general_kmeans_model = KMeans(n_clusters = 8, random_state = 1)
good_columns = per_game_set._get_numeric_data().dropna()
good_columns.drop(columns = ["Age", "G", "GS", "G", "GS", "MP", "FT%"])
general_kmeans_model.fit(good_columns)
labels = general_kmeans_model.labels_

LeBron = good_columns.loc[ per_game_set['Player'] == 'LeBron James',: ]
Davis = good_columns.loc[ per_game_set['Player'] == 'Anthony Davis',: ]

LeBron_list = LeBron.values.tolist()
LeBron_label = general_kmeans_model.predict(LeBron_list)

Davis_list = Davis.values.tolist()
Davis_label = general_kmeans_model.predict(Davis_list)

print("Lebron Grouping: {}".format(LeBron_label))
print("Davis Grouping: {}".format(Davis_label))

print("\nTop Tier of Players in NBA\n")

for index, row in per_game_set.iterrows():
    name = row['Player']
    column_row = good_columns.loc[ per_game_set['Player'] == name,: ]

    row_list = column_row.values.tolist()
    row_label = general_kmeans_model.predict(row_list)

    if row_label[0] == 6:
        print("{}".format(name))

## Model for Shooter Classification ##

advanced_stat_set = pd.read_csv("archive/nba_2020_advanced.csv")

summative_stat_set = pd.merge(advanced_stat_set, per_game_set, on = "Player")

summative_stat_set.drop_duplicates(subset = "Player", keep = False, inplace = True)

shooting_kmeans_model = KMeans(n_clusters = 5, random_state = 1)
shooting_good_columns = summative_stat_set[['3P%', 'eFG%', '3PA', 'TS%', '3PAr']]._get_numeric_data().dropna()
shooting_kmeans_model.fit(shooting_good_columns)
labels = shooting_kmeans_model.labels_

print("\nTop Tier of Shooters in NBA\n")

for index, row in summative_stat_set.iterrows():
    name = row['Player']
    column_row = shooting_good_columns.loc[ summative_stat_set['Player'] == name,: ]

    row_list = column_row.values.tolist()
    row_label = shooting_kmeans_model.predict(row_list)

    if row_label[0] == 2:
        print("{}".format(name))

