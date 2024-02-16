from pytube import YouTube
import cv2
import os
from tqdm import tqdm

def progress_bar(stream, chunk, bytes_remaining):
    total_size = stream.filesize
    bytes_downloaded = total_size - bytes_remaining
    percentage_of_completion = bytes_downloaded / total_size * 100
    pbar.update(bytes_downloaded - pbar.n)  # Update tqdm progress bar accordingly


video_url = "https://youtube.com/shorts/9AptJC_ahEk?si=3ag1EXE80xgP5hSX"
video_url = "https://www.youtube.com/shorts/5sUhFYATYeM"
video_url = "https://www.youtube.com/shorts/Pa-TEaFoCK4"
video_url = "https://youtube.com/shorts/Nr_J52v2_54?si=mX8hT62d8RvHIDiH"


# Initialize tqdm progress bar
pbar = tqdm(total=100, unit='B', unit_scale=True, desc="Downloading Video")

yt = YouTube(video_url, on_progress_callback=progress_bar)



stream = yt.streams.get_highest_resolution()

# finding the free video name (number)
# VIDEOS_DIR = os.path.join('.', 'loaded_videos')
VIDEOS_DIR = "C:\\Users\\micro\\Desktop\\Yolo\\loaded_videos"
os.chdir(VIDEOS_DIR)
num = 1
while True:
    # name = os.path.join(VIDEOS_DIR, f"{num}.mp4")
    name = f"{num}.mp4"
    if os.path.exists(name):
        num += 1
    else:
        stream.download(output_path=VIDEOS_DIR, filename = name)
        break

pbar.close()
