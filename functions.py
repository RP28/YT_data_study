import re
import numpy as np
import requests
from PIL import Image
import numpy as np
from io import BytesIO
from googleapiclient.discovery import build


def _convert_duration_to_minutes(duration):
    hours = 0
    minutes = 0
    seconds = 0
    hours_match = re.search(r"(\d+)H", duration)
    minutes_match = re.search(r"(\d+)M", duration)
    seconds_match = re.search(r"(\d+)S", duration)
    if hours_match:
        hours = int(hours_match.group(1))
    if minutes_match:
        minutes = int(minutes_match.group(1))
    if seconds_match:
        seconds = int(seconds_match.group(1))
    total_minutes = hours * 60 + minutes + seconds / 60
    return total_minutes


def find_outperformed_videos(data, threshold=1.5):
    q3 = np.percentile(data, 75)
    iqr = q3 - np.percentile(data, 25)
    upper_threshold = q3 + threshold * iqr
    outlier_indices = [
        index for index, value in enumerate(data) if value > upper_threshold
    ]
    return outlier_indices


def url_to_numpy_array(url):
    response = requests.get(url)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        img_array = np.array(img)
        return img_array
    else:
        print("Failed to fetch image")
        return None


API_KEY = "API_KEY"
API_SERVICE_NAME = "youtube"
API_VERSION = "v3"

youtube = build(API_SERVICE_NAME, API_VERSION, developerKey=API_KEY)


def get_channel_stats(channel_ids):
    request = youtube.channels().list(
        part="snippet,contentDetails,statistics", id=",".join(channel_ids)
    )
    response = request.execute()

    all_data = []
    for i in range(len(response["items"])):
        data = {
            "channel_name": response["items"][i]["snippet"]["title"],
            "subscribers": response["items"][i]["statistics"]["subscriberCount"],
            "views": response["items"][i]["statistics"]["viewCount"],
            "total_videos": response["items"][i]["statistics"]["videoCount"],
            "playlist_id": response["items"][i]["contentDetails"]["relatedPlaylists"][
                "uploads"
            ],
        }
        all_data.append(data)

    return all_data


def get_video_ids(playlist_id):
    request = youtube.playlistItems().list(
        part="contentDetails", playlistId=playlist_id, maxResults=50
    )
    response = request.execute()

    video_ids = []
    for i in range(len(response["items"])):
        video_ids.append(response["items"][i]["contentDetails"]["videoId"])

    next_page_token = response.get("nextPageToken")

    while next_page_token is not None:
        request = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page_token,
        )
        response = request.execute()

        for i in range(len(response["items"])):
            video_ids.append(response["items"][i]["contentDetails"]["videoId"])

        next_page_token = response.get("nextPageToken")

    return video_ids


def get_video_details(video_ids):
    all_video_details = []
    for i in range(0, len(video_ids), 50):
        request = youtube.videos().list(
            part="snippet,statistics,contentDetails", id=",".join(video_ids[i : i + 50])
        )
        response = request.execute()

        for video in response["items"]:
            video_detail = {
                # "title": video["snippet"]["title"],
                "published_date": video["snippet"]["publishedAt"],
                "tumbnail_url": video["snippet"]["thumbnails"]["high"]["url"],
                "category_id": video["snippet"]["categoryId"],
                "views": video["statistics"].get("viewCount"),
                "likes": video["statistics"].get("likeCount"),
                "comments_count": video["statistics"].get("commentCount"),
                "video_duration": _convert_duration_to_minutes(
                    video["contentDetails"]["duration"]
                ),
            }
            all_video_details.append(video_detail)

    return all_video_details


def get_channels(search_query):
    channel_ids = []

    request = youtube.search().list(
        part="snippet",
        q=search_query,
        type="channel",
        maxResults=50,
    )
    response = request.execute()

    for item in response["items"]:
        channel_ids.append(item["id"]["channelId"])

    channel_ids = list(set(channel_ids))
    return channel_ids
