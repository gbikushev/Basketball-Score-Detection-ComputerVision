from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.system("yolo task=detect mode=train model=yolov8n.pt data=config.yaml epochs=25 plots=True")