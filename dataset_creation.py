from functions import (
    get_channel_stats,
    get_video_details,
    get_video_ids,
    find_outperformed_videos,
    get_channels,
)
import csv

c_ids = get_channels("science")[:10]
dataset = []

channel_details = get_channel_stats(c_ids)
for c_detail in channel_details:
    channel_name = c_detail["channel_name"]
    temp_dataset = []
    views = []
    likes = []
    comments = []
    video_ids = get_video_ids(c_detail["playlist_id"])
    all_video_details = get_video_details(video_ids)
    for video_detail in all_video_details:
        if (
            video_detail["video_duration"] > 4
            and int(video_detail["published_date"][:4]) > 2016
            and video_detail["views"] is not None
            and video_detail["likes"] is not None
            and video_detail["comments_count"] is not None
        ):
            views.append(int(video_detail["views"]))
            likes.append(int(video_detail["likes"]))
            comments.append(int(video_detail["comments_count"]))
            temp_dataset.append(
                {
                    "thumbnail": video_detail["tumbnail_url"],
                    "views": int(video_detail["views"]),
                    "likes": int(video_detail["likes"]),
                    "comments_count": int(video_detail["comments_count"]),
                    "views_wise_outperformed": 0,
                    "likes_wise_outperformed": 0,
                    "comments_wise_outperformed": 0,
                    "channel_name": channel_name,
                }
            )
    views_wise_outperformed_index = find_outperformed_videos(views)
    likes_wise_outperformed_index = find_outperformed_videos(likes)
    comments_wise_outperformed_index = find_outperformed_videos(comments)

    for idx in views_wise_outperformed_index:
        temp_dataset[idx]["views_wise_outperformed"] = 1
    for idx in likes_wise_outperformed_index:
        temp_dataset[idx]["likes_wise_outperformed"] = 1
    for idx in comments_wise_outperformed_index:
        temp_dataset[idx]["comments_wise_outperformed"] = 1

    dataset.extend(temp_dataset)
    print(channel_name + " is completed.")

print("Dataset acquired.")

csv_filename = "dataset2.csv"

field_names = dataset[0].keys()

with open(csv_filename, mode="w", newline="") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=field_names)
    writer.writeheader()
    writer.writerows(dataset)

print(f"CSV file '{csv_filename}' created.")
