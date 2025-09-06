# Calculate Elbow Point for Optimal k Value App: Project Overview
- Built a song recommendation engine based on users selected artist. Example - Michael Jackson
- Optimized ``K-Nearest Neighbors`` algorithm to recommend top songs that match user's input and its genre
- Creates a CSV file for recommended artists based on inout. Default to first 50 songs.
- Calculate Elbow Point for Optimal k Value for music search based on song popularity.
- Build the Clustering Model and Calculating Distortion and Inertia
- Fit the K-means model for different values of k (number of clusters) and 
- Calculate both the distortion and inertia for each value.
## Code and Resources Used
**Python Version:** 3.13 <br>
**Packages:** pandas, numpy, sklearn, matplotlib <br>
**Data Source:** https://www.kaggle.com/datasets/saurabhshahane/spotgen-music-dataset <br>

#Reference:
#https://www.geeksforgeeks.org/machine-learning/elbow-method-for-optimal-value-of-k-in-kmeans/

Provide artist name as argument. Default to artists ‘Eagles’
The application will take the artist’s name and find his/her genre 
Optimized K-Nearest Neighbors algorithm to recommend top songs that match user artists given genre
song popularity, acousticness, danceability, energy, instrumentalness, valence, tempo variables
 
> % python spotify_music_elbow.py --artist 'Simon & Garfunkel'

> % python spotify_music_elbow.py --artist ‘Michael Jackson’

> % python spotify_music_elbow.py --artist ‘Taylor Swift’

 



 






