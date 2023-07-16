import cv2
import tensorflow as tf
import numpy as np
import time
import random
import spotipy
import webbrowser
from spotipy.oauth2 import SpotifyClientCredentials
from googleapiclient.discovery import build

class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

model = tf.keras.models.load_model("./ferNetModel.h5")

faceDetect = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

video = cv2.VideoCapture(0)
capture_time = time.time() + 10  # Set the initial capture time

# Spotify client credentials
client_id = '47c52f0f93basddasdsa7443bb45e266ad959f7fsdas9'
client_secret = '0b61ccfc898746eeab46f474dsadad4712b9ddwwsdasd'

# Spotify authentication
client_credentials_manager = SpotifyClientCredentials(client_id, client_secret)
spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


def spotifyApi(emotion_label):
    playlists = spotify.search(q=emotion_label, type='playlist')['playlists']['items']

    if playlists:
        random_playlist = random.choice(playlists)
        playlist_name = random_playlist['name']
        playlist_uri = random_playlist['uri']
        playlist_url = random_playlist['external_urls']['spotify']
        print("Recommended Playlist:", playlist_name)
        print("Playlist URI:", playlist_uri)
        print("Playlist URL:", playlist_url)

        # Play the playlist in the default web browser
        webbrowser.open(playlist_url)
        # play_playlist(playlist_uri)


# Function to play the playlist using Spotify Web API
def play_playlist(playlist_uri):
    spotify.start_playback(context_uri=playlist_uri)


# Set up the YouTube Data API client
api_key = 'AIzaSyAivg6ZTH_BO9kd9ZeMy4nJ5WmjMy8nMwQ'
youtube = build('youtube', 'v3', developerKey=api_key)


# Function to search and play YouTube videos based on the detected emotion
def search_and_play_video(query):
    # Call the search.list method to retrieve search results
    search_response = youtube.search().list(
        part='snippet',
        q=query + '+songs',
        type='video',
        maxResults=1
    ).execute()

    # Process the search response
    if 'items' in search_response:
        video_id = search_response['items'][0]['id']['videoId']
        title = search_response['items'][0]['snippet']['title']
        print(f"Title: {title}")
        print(f"Video ID: {video_id}")
        print("----------")

        # Open the video in a web browser
        video_url = f"https://www.youtube.com/watch?v={video_id}&autoplay=1"
        webbrowser.open(video_url)


while True:
    ret, frame = video.read()

    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)

    for x, y, w, h in faces:
        sub_face_img = gray[y: y + h, x: x + w]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized / 255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        emotion_label = class_names[label]
        print(emotion_label)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame, class_names[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Search and play YouTube videos based on the detected emotion label
        search_and_play_video(emotion_label)

        # Retrieve the playlist based on the detected emotion label
        # emotion_label = class_names[label]
        # spotifyApi(emotion_label)

    cv2.imshow("Frame", frame)

    # Check for keyboard input
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

    # Capture image every minute
    current_time = time.time()
    if current_time >= capture_time:
        cv2.imwrite("captured_image.jpg", frame)
        capture_time += 10  # Update the capture time by adding 60 seconds

    # Wait for 15 seconds
    time.sleep(15)

video.release()
cv2.destroyAllWindows()
