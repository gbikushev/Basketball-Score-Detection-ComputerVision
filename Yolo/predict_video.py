import os

from ultralytics import YOLO
import cv2


VIDEOS_DIR = os.path.join('.', 'videos')

video_path = os.path.join(VIDEOS_DIR, 'basket3.mp4')
video_path_out = '{}_out10.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join('.', 'runs', 'detect', 'train_long', 'weights', 'best.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5

while ret:

    results = model(frame)[0]
    
    for result in results.boxes.data.tolist():
        # print(result)
        x1, y1, x2, y2, score, class_id = result
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), colors[int(class_id)], 4)
            text = f"{str(results.names[int(class_id)])} ({score:.1f})"
            cv2.putText(frame, text.upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, colors[int(class_id)], 3, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()
    # break

cap.release()
out.release()
cv2.destroyAllWindows()
