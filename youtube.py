import sys
from googleapiclient.discovery import build
import webbrowser


def youtube(query):
    print("Emotion: ", query)
    # Set up the YouTube Data API client
    api_key = 'AIzaSyAcspTKj8udvDMoZMIdvx-oyMbSJb6ysnY'
    yt = build('youtube', 'v3', developerKey=api_key)

    search_response = yt.search().list(
        part='snippet',
        q=query + '+songs',
        type='video',
        maxResults=1
    ).execute()

    # Process the search response
    if 'items' in search_response:
        video_id = search_response['items'][0]['id']['videoId']
        title = search_response['items'][0]['snippet']['title']
        # Open the video in a web browser
        video_url = f"https://www.youtube.com/watch?v={video_id}&autoplay=1"

        print("\nYOUTUBE")
        print(f"Video Title: {title}")
        print(f"Video ID: {video_id}")
        print(f"Video URL: {video_url}")
        print("----------")

        webbrowser.open(video_url)


if __name__ == "__main__":
    youtube(sys.argv[1])
