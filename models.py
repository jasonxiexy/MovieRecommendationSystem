import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import openpyxl
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


# importing the data files and merge to the format as we needed
column_name = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv("file.tsv", sep='\t', names=column_name)
df.head()

movie_titles = pd.read_csv('Movie_Id_Titles.csv')
movie_titles.head()

data = pd.merge(df, movie_titles, on='item_id')
data.head()

movie_average_ratings = data.groupby('title')['rating'].mean().sort_values(ascending=False).reset_index().rename(columns={'rating':'Average Rating'})
# print(movie_average_ratings.head(5))

movie_rating_count = data.groupby('title')['rating'].count().sort_values(ascending=False).reset_index().rename(columns={'rating':'Rating Count'})
# print(movie_rating_count.head())

movies_rating_count_avg = pd.merge(movie_rating_count, movie_average_ratings, on='title')
# print(movies_rating_count_avg.head())



# Visualizing the data in different angles
# first histogram of
sns.set_style('white')
plt.hist(movies_rating_count_avg['Rating Count'], bins=80, color='tab:purple')
plt.ylabel('Ratings Count(Scaled)', fontsize=16)
plt.savefig('ratingcounthist.jpg')
plt.clf()

# histogram 2 shows the distribution of different ratings values
sns.set_style('white')
plt.hist(movies_rating_count_avg['Average Rating'], bins=80, color='tab:purple')
plt.ylabel('Average Rating', fontsize=16)
plt.savefig('avgratinghist.jpg')
plt.clf()

# histogram 3 shows the partial amount of movies with higher ratings have considerable amount of ratings
plot=sns.jointplot(x='Average Rating', y='Rating Count', data=movies_rating_count_avg, alpha=0.5, color='tab:pink')
plot.savefig('joinplot.jpg')

# Eliminating the outliers
rating_with_RatingCount = data.merge(movie_rating_count, left_on='title', right_on='title', how='left')
rating_with_RatingCount.head()

pd.set_option('display.float_format', lambda x: '%.3f' % x)
# print(rating_with_RatingCount['Rating Count'].describe())
# the statistics about the number of user ratings on movie
# count   100003.000
# mean       169.106
# std        122.220
# min          1.000
# 25%         72.000
# 50%        146.000
# 75%        240.000
# max        584.000

# Finding the popular movies if the number of ratings is >= 50
popularity_threshold = 50
popular_movies = rating_with_RatingCount[rating_with_RatingCount['Rating Count'] >= popularity_threshold]
popular_movies_list = pd.DataFrame(popular_movies).groupby('title', group_keys=True)


# list the total number of ratings with respect to each movie, and save to Excel file
movie_features_df = popular_movies.pivot_table(index='title', columns='user_id', values='rating').fillna(0)
movie_features_df.head()
# print(movie_features_df.shape[0])
movie_features_df.to_excel('output.xlsx')


# Implementing the KNN model to train the data
# using cosine metric to determine the distance between its neighbors
movie_features_df_matrix = csr_matrix(movie_features_df.values)
model_knn = NearestNeighbors(metric= 'cosine', algorithm= 'brute')
model_knn.fit(movie_features_df_matrix)

# print the popular movie list for users to select and be recommended.
pd.set_option('display.max_colwidth', None)
print(f"Below is the popular movie list, and also with a auto-generated csv file (popular_movies.csv) to check the full list.\n\n")
print(pd.DataFrame(popular_movies_list.first().reset_index()))
popular_movies_list.first().reset_index().to_csv('popular_movies.csv')

movie_id = int(input("Please choose the index (first column) of your movie [0-{n}]: ".format(n=605-1)))

# query_index = np.random.choice(movie_features_df.shape[0])
# print(query_index)
distances, indices = model_knn.kneighbors(movie_features_df.iloc[movie_id, :].values.reshape(1, -1), n_neighbors= 6)

for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}:\n'.format(movie_features_df.index[movie_id]))
    else:
        print('{0}: {1}, with distance of {2}:'.format(i, movie_features_df.index[indices.flatten()[i]], distances.flatten()[i]))
