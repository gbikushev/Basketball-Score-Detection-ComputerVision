from fastapi import HTTPException, status
from fastapi import UploadFile
from moviepy.editor import VideoFileClip
import moviepy.editor as mp
import moviepy.config as mp_conf
import os

# check the file format
def check_file_format(filename):
    allowed_format = 'mp4'
    file_ext = filename[filename.rfind('.'):].lower()
    if file_ext == f'.{allowed_format}':
        return True
    else:
        return False

# saves the video to disk
async def save_video(video: UploadFile, video_path: str) -> None:
    with open(video_path, "wb") as video_file:
        video_file.write(await video.read())


# saves the video in the specified directory
async def prepare_video_files(Video):
    UPLOADS_DIR = "result_videos"
    os.makedirs(UPLOADS_DIR, exist_ok=True)

    video_path = os.path.join(UPLOADS_DIR, Video.filename)
    await save_video(Video, video_path)

    processed_filename = f"processed_{Video.filename}"
    processed_path = os.path.join(UPLOADS_DIR, processed_filename)

    return video_path, processed_path

# change the result video fps
def change_fps(processed_video_path):

    result_video_path = processed_video_path.replace("processed_", "result_")
    # Desired frame rate
    desired_frame_rate = 30
    # Load the video
    clip = VideoFileClip(processed_video_path)
    # Set the new frame rate
    clip = clip.set_fps(desired_frame_rate)
    # Write the result to a file
    clip.write_videofile(result_video_path, codec='libx264')
    return result_video_path


# calculates the size of the video
async def calculate_video_size(file_path):
    try:
        size_in_bytes = os.path.getsize(file_path)
        return int(size_in_bytes)
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Файл не найден")
