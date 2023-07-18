import random
import spotipy
import webbrowser
import sys
from spotipy.oauth2 import SpotifyClientCredentials


def spotifyApi(emotion_label):

    print("Emotion: ", emotion_label)

    # Spotify client credentials
    client_id = '<your_client_id>'
    client_secret = '<your_client_secret>'

    # Spotify authentication
    client_credentials_manager = SpotifyClientCredentials(client_id, client_secret)
    spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    # The search() method of the Spotify API client is called to search for playlists based on the 
    # emotion_label and retrieve the search response.
    playlists = spotify.search(q=emotion_label, type='playlist')['playlists']['items']

    # If the search response contains any playlists, a random playlist is chosen from the search results.
    if playlists:
        random_playlist = random.choice(playlists)
        playlist_name = random_playlist['name']
        playlist_uri = random_playlist['uri']
        playlist_url = random_playlist['external_urls']['spotify']
        print("\nSPOTIFY")
        print("Recommended Playlist:", playlist_name)
        print("Playlist URI:", playlist_uri)
        print("Playlist URL:", playlist_url)

        # Open the playlist in the default web browser
        webbrowser.open(playlist_url)


if __name__ == "__main__":
    spotifyApi(sys.argv[1])
