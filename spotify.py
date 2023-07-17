import random
import spotipy
import webbrowser
import sys
from spotipy.oauth2 import SpotifyClientCredentials


def spotifyApi(emotion_label):
    print("Emotion: ", emotion_label)
    # Spotify client credentials
    client_id = '47c52f0f93b7443bb45e266ad959f7f9'
    client_secret = '0b61ccfc898746eeab46f474d4712b9d'

    # Spotify authentication
    client_credentials_manager = SpotifyClientCredentials(client_id, client_secret)
    spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    playlists = spotify.search(q=emotion_label, type='playlist')['playlists']['items']

    if playlists:
        random_playlist = random.choice(playlists)
        playlist_name = random_playlist['name']
        playlist_uri = random_playlist['uri']
        playlist_url = random_playlist['external_urls']['spotify']
        print("\nSPOTIFY")
        print("Recommended Playlist:", playlist_name)
        print("Playlist URI:", playlist_uri)
        print("Playlist URL:", playlist_url)

        # Play the playlist in the default web browser
        webbrowser.open(playlist_url)


if __name__ == "__main__":
    spotifyApi(sys.argv[1])
