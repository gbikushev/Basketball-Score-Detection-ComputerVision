import os
import cv2

VIDEOS_DIR = "C:\\Users\\micro\\Desktop\\Yolo\\loaded_videos"
os.chdir(VIDEOS_DIR)

file_names = os.listdir('.')
for file in file_names:
    SAVE_DIR = "C:\\Users\\micro\\Desktop\\Yolo\\cutted_videos"
    os.chdir(SAVE_DIR)
    dir_name = file.replace('.mp4', '')
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        print(f"Directory '{dir_name}' was created.")
        SAVE_DIR = os.path.join(SAVE_DIR, dir_name)
    else:
        print(f"Directory '{dir_name}' already exists.")
        continue

    video_path = os.path.join(VIDEOS_DIR, file)
    cap = cv2.VideoCapture(video_path)

    ret, frame = cap.read()
    H, W, _ = frame.shape

    frame_count = 0
    while ret:
        frame_count += 1

        frame_file = os.path.join(SAVE_DIR, f"frame{frame_count:04d}.jpg")
        cv2.imwrite(frame_file, frame)


        ret, frame = cap.read()
