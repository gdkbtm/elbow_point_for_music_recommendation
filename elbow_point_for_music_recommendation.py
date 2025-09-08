# Import libraries
import glob
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # or 'Qt5Agg'
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 
from scipy.spatial.distance import cdist
import argparse

import os
import re
import sys

# Define list of genres and audio features of songs for audience to choose from
audio_feats = ["acousticness", "danceability", "energy", "instrumentalness", "valence", "tempo"]

def read_data():
    #global df  # Declare df as global within the function
    all_files = glob.glob("csv_data/*.csv")
    list_contact_csv = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        #add all csv data
        list_contact_csv.append(df) 
    df = pd.concat(list_contact_csv, axis=0, ignore_index=True)

    return df

def load_data(name):
    filtered_df = []
    message = 1

    df = read_data()
    #if df is not None:
    df = df.drop_duplicates(subset=['uri'], keep='first')
    #add new column
    if(len(name) > 0):
        df['artists_name_lower'] = df['artists_name'].str.lower()
    
        filtered_df = df[df['artists_name_lower'] == name.lower()]
        if(len(filtered_df) == 0):
            message = 0
        df['genres'] = df.genres.apply(lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])
    exploded_track_df = df.explode("genres")

    return exploded_track_df, filtered_df, message

def find_highest_duplicate(arr):
    counts = Counter(arr)
    duplicates = [item for item, count in counts.items() if count > 1]
    if duplicates:
        return max(duplicates)
    else:
        return None
    
def get_artist_genre(highest_dup):
    first_token = highest_dup.split(',')
    if len(first_token) > 1:
        highest_dup = str(highest_dup)[1:-1]
        highest_dup = highest_dup.split(',')[0]
        highest_dup = str(highest_dup)[1:-1]
    else:
        highest_dup = str(highest_dup)[2:-2]
    return highest_dup;


# Define function to return Spotify URIs and audio feature values of top neighbors (ascending)
def n_neighbors_uri_audio(exploded_track_df, filtered_df, artist_select, start_year, end_year, test_feat):
    # The artist given
    if(len(artist_select) > 0):
        print('The artist given: ', artist_select)
        print('The found in csv files: ')
        genre = find_highest_duplicate(filtered_df.genres)
        #genre_new = str(genre_new)[2:-2]
        #get the genre string value
        genre = get_artist_genre(genre)
        print('The genre of the given artist: ', genre)
        print('The start year: ', min(filtered_df.release_year))
        print('The end year: ', max(filtered_df.release_year))
        #test_feat = [acousticness, danceability, energy, instrumentalness, valence, tempo]
        print('The acousticness: ', min(filtered_df.acousticness))
        print('The acousticness: ', max(filtered_df.acousticness))
        print('The danceability: ', min(filtered_df.danceability))
        print('The danceability: ', max(filtered_df.danceability))
        print('The energy: ', min(filtered_df.energy))
        print('The energy: ', max(filtered_df.energy))
        print('The instrumentalness: ', min(filtered_df.instrumentalness))
        print('The instrumentalness: ', max(filtered_df.instrumentalness))
        print('The valence: ', min(filtered_df.valence))
        print('The valence: ', max(filtered_df.valence))
        print('The tempo: ', min(filtered_df.tempo))
        print('The tempo: ', max(filtered_df.tempo))

    if(len(artist_select) == 0):
        genre_data = exploded_track_df[(exploded_track_df["genres"]==genre) & (exploded_track_df["release_year"]>=start_year) & (exploded_track_df["release_year"]<=end_year)]
    else:
        genre_data = exploded_track_df[(exploded_track_df["genres"]==genre) & (exploded_track_df["release_year"]>=min(filtered_df.release_year)) & (exploded_track_df["release_year"]<=max(filtered_df.release_year))]
        #calculate test_feat from given attributes 
        #get the max and and min for the artist and get the difference 
        acousticness = (max(filtered_df.acousticness) - min(filtered_df.acousticness))/2
        danceability = (max(filtered_df.danceability) - min(filtered_df.danceability))/2
        energy = (max(filtered_df.energy) - min(filtered_df.energy))/2
        instrumentalness = (max(filtered_df.instrumentalness) - min(filtered_df.instrumentalness))/2
        valence = (max(filtered_df.valence) - min(filtered_df.valence))/2
        tempo = (max(filtered_df.tempo) - min(filtered_df.tempo))/2
        test_feat = [acousticness, danceability, energy, instrumentalness, valence, tempo]
    genre_data = genre_data.sort_values(by='popularity', ascending=False)[:500] # use only top 500 most popular songs
    #print(len(genre_data))
    neigh = NearestNeighbors()
    neigh.fit(genre_data[audio_feats].to_numpy())
    
    n_neighbors = neigh.kneighbors([test_feat], n_neighbors=len(genre_data), return_distance=False)[0]
    
    #Search nearest neighbor
    artists_id = genre_data.iloc[n_neighbors]['artists_id'].to_numpy()    
    artists_name = genre_data.iloc[n_neighbors]['artists_name'].to_numpy()
    artist_info = [genre_data.iloc[n_neighbors]['artists_id'].to_numpy(), genre_data.iloc[n_neighbors]['artists_name'].to_numpy()]
    uris = genre_data.iloc[n_neighbors]["uri"].tolist()
    audios = genre_data.iloc[n_neighbors][audio_feats].to_numpy()
          
    return genre, genre_data, uris, audios, artists_id, artists_name, artist_info

#creates artist_recommend.csv for recommeneded songs 
def create_artist_recommend(genre_data):
    temp_folder_path = 'music_data'
    
    artist_recommend_file = 'artist_recommend.csv'
    
    #number of songs recommended based on the artist given. 
    num_pred = 50

    genre_data[["artists_name", "name", "genres", "release_year", "popularity"]][:num_pred].to_csv(os.path.join(temp_folder_path, artist_recommend_file),
                                   index=False)
    print("Artists recommend songs file ", artist_recommend_file, " created.")
    
#The Elbow Point: Optimal k Value
#find optimal k value for popularity vs loudness
def find_elbowpoint():
    y_var = [genre_data.acousticness, genre_data.danceability, genre_data.energy, genre_data.instrumentalness, genre_data.valence, genre_data.tempo]
  
    for item1, item2 in zip(audio_feats, y_var):
        x = genre_data.popularity
        y = item2
        print('item1 ', item1)
        print('item2 ', item2)
        X = np.array(list(zip(x, y))).reshape(len(x), 2)
        plt.title(f'Recommended Songs for {genre} genre')
        plt.xlabel('Popularity')
        plt.ylabel(item1)
        plt.scatter(x, y)
        plt.show()
    print('X ', X)
    return X;

#Building the Clustering Model and Calculating Distortion and Inertia
#fit the K-means model for different values of k (number of clusters) and 
#calculate both the distortion and inertia for each value.
def cluster_model(X):
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    K = range(1, 10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k, random_state=42).fit(X)
        
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'Minkowski'), axis=1)**2) / X.shape[0])
        inertias.append(kmeanModel.inertia_)
        
        mapping1[k] = distortions[-1]
        mapping2[k] = inertias[-1]

    return mapping1, distortions, K

#Displaying Distortion Values
def distoted_values(mapping1, distortions, K):  
    print("Distortion values:")
    for key, val in mapping1.items():
        print(f'{key} : {val}')

    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method using Distortion')
    plt.show()

#Displaying Inertia Values
def inertia_values():    
    k_range = range(1, 5)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        y_kmeans = kmeans.fit_predict(X)
        
        plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', marker='o', edgecolor='k', s=100)
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                    s=300, c='red', label='Centroids', edgecolor='k')
        plt.title(f'K-means Clustering (k={k})')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    #args.add_argument("--genre", "-g", type=str, default='rock')
    args.add_argument("--artist", "-a", type=str, default='Eagles')

    parsed_args = args.parse_args()
    given_artist = parsed_args.artist 
    # Load data
    exploded_track_df, filtered_df, message = load_data(given_artist)
    if(message == 0):
        print('The given artist', given_artist, 'not found. Please try again.')
    else:
        test_feat = [0.5, 0.5, 0.5, 0.0, 0.45, 118.0] # not used
        start_year = 1960
        end_year = 2000
        genre, genre_data, uris, audios, artists_id, artists_name, artist_info = n_neighbors_uri_audio(exploded_track_df, filtered_df, parsed_args.artist, start_year, end_year, test_feat)

        create_artist_recommend(genre_data)
        #The Elbow Point: Optimal k Value
        #find optimal k value for popularity vs loudness
        X = find_elbowpoint()
        #Building the Clustering Model and Calculating Distortion and Inertia
        #fit the K-means model for different values of k (number of clusters) and 
        #calculate both the distortion and inertia for each value.
        mapping1, distortions, K = cluster_model(X)
        #Displaying Distortion Values
        distoted_values(mapping1, distortions, K)
        #Displaying Inertia Values
        inertia_values()

