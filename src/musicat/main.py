import spotipy
import pandas as pd
import numpy as np
from spotipy.oauth2 import SpotifyOAuth
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

YOUR_APP_CLIENT_ID = "b33aeb17d51c4b61bcc3c3522b3892bc"
YOUR_APP_CLIENT_SECRET = "6a1722d7cba44a2182513403c6286a91"
YOUR_APP_REDIRECT_URI = "http://localhost:1234"

spotify = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=YOUR_APP_CLIENT_ID,
                                               client_secret=YOUR_APP_CLIENT_SECRET,
                                               redirect_uri=YOUR_APP_REDIRECT_URI,
                                               scope="user-library-read"))

df = pd.read_csv('/Users/wenhanyang/Desktop/echo_music_league.csv')
df = df.fillna(0)
results = spotify.audio_features(df.iloc[0,1:])
row, col = df.shape
print("row: ", row, "col: ", col)
total = 0
for r in range(1, row):
    print("name: ", df.iloc[r, 0])
    features = None
    score = []
    for c in range(1, col):
        f = [i for i in list(results[c-1].values()) if type(i) == float or type(i) == int]
        f = np.array(f)
        if features is None:
            features = f
        else:
            features = np.vstack((features, f))
        score.append(df.iloc[r, c])
    score = np.array(score)
    X_train, X_test, y_train, y_test = train_test_split(features, score, test_size = 0.1, random_state = 53)
    regr = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    print(regr.score(X_train,y_train))
    print(regr.score(X_test,y_test))

    url = 'https://open.spotify.com/track/4svYLhfpY2FC0lfADH3y0t?si=bf924a7e90614f59'
    result = spotify.audio_features(url)
    f = [i for i in list(result[0].values()) if type(i) == float or type(i) == int]
    f = np.array(f)
    f = f.reshape(1,-1)
    score = regr.predict(f)
    print("give you score: ", score)
    total += float(score.item())
print("final score: ", total)

