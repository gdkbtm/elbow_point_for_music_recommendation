# Song Recommendation App: Project Overview
- Built a song recommendation engine based on users' selection of song features (e.g. genres, acousticness, danceability, etc.)
- Optimized ``K-Nearest Neighbors`` algorithm to recommend top songs that match user preferences
- Designed app layout and user interface using Streamlit

## Code and Resources Used
**Python Version:** 3.13 <br>
**Packages:** pandas, numpy, sklearn, streamlit <br>
**Data Source:** https://www.kaggle.com/datasets/saurabhshahane/spotgen-music-dataset <br>
**Web Framework Requirements:** ``pip install -r requirements.txt`` <br>


#Reference:
#https://www.geeksforgeeks.org/machine-learning/elbow-method-for-optimal-value-of-k-in-kmeans/

Provide artist name as argument. Default to artists ‘Eagles’
The application will take the artist’s name and find his/her genre 
Optimized K-Nearest Neighbors algorithm to recommend top songs that match user artists given genre
song popularity, acousticness, danceability, energy, instrumentalness, valence, tempo variables
 
>spotity_find_elbow_point % python spotify_music_elbow.py --artist 'Simon & Garfunkel'

>spotity_find_elbow_point % python spotify_music_elbow.py --artist ‘Michael Jackson’

>spotity_find_elbow_point % python spotify_music_elbow.py --artist ‘Taylor Swift’

Calculate Elbow Point for Optimal k Value for music search based on song popularity.

Build the Clustering Model and Calculating Distortion and Inertia
Fit the K-means model for different values of k (number of clusters) and 
Calculate both the distortion and inertia for each value.


 






