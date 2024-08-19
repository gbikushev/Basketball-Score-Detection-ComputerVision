import numpy as np
import supervision as sv
import os
from ultralytics import YOLO
from supervision.draw.color import Color
from supervision.draw.utils import draw_text
from src.process_video.classes import PolygonZone, PolygonZoneAnnotator, bounding_box_annotator, label_annotator, rectangles_intersect
# from classes import PolygonZone, PolygonZoneAnnotator, bounding_box_annotator, label_annotator, rectangles_intersect
import tqdm


def detect(frame: np.ndarray, model) -> sv.Detections:
    result = model.predict(frame, imgsz=1280, conf=0.5, verbose=False)[0]
    return sv.Detections.from_ultralytics(result)


def setup_zone_above(detections, frame_wh):
    zones = []
    count_zone = 0
    for i in range(len(detections)):
        obj = detections[i]

        # choosing only baskets with confidence score >= 0.5
        if (obj.data.get('class_name')[0] == 'basket') & (obj.confidence[0] >= 0.5):
            count_zone += 1

            obj_xyxy = obj.xyxy[0]

            # get coordinates of 4 points of bounding box
            obj_p1 = np.array([obj_xyxy[0], obj_xyxy[1]])
            obj_p2 = np.array([obj_xyxy[0], obj_xyxy[3]])
            obj_p3 = np.array([obj_xyxy[2], obj_xyxy[3]])
            obj_p4 = np.array([obj_xyxy[2], obj_xyxy[1]])

            # get height and width of bounding box
            obj_height = obj_xyxy[3] - obj_xyxy[1]
            obj_width = obj_xyxy[2] - obj_xyxy[0]

            # polygon construction strategy
            # 1) polygon width will be width of "basket" bounding box increased on 10% (5% from left and right sides)
            # 2) polygon height will be height of "basket" bounding box
            # 3) obj_p1 and obj_p4 will become the zone_p2 and zone_p3 respectively
            # 4) polygon will be exactly above the basket bounding box

            zone_p1 = [obj_p1[0] - 0.05 * obj_width, obj_p1[1] - obj_height]
            zone_p2 = [obj_p1[0] - 0.05 * obj_width, obj_p1[1]]
            zone_p3 = [obj_p4[0] + 0.05 * obj_width, obj_p4[1]]
            zone_p4 = [obj_p4[0] + 0.05 * obj_width, obj_p4[1] - obj_height]

            # round coordinates of each point of zone
            zone_p1 = list(map(round, zone_p1))
            zone_p2 = list(map(round, zone_p2))
            zone_p3 = list(map(round, zone_p3))
            zone_p4 = list(map(round, zone_p4))

            zone_coordinates = np.array([zone_p1, zone_p2, zone_p3, zone_p4])
            zone_name = f"Zone {count_zone}"

            zone = {
                'name': zone_name,
                'polygon': zone_coordinates,
                'count': 0
            }

            zone['PolygonZone'] = PolygonZone(
                polygon=zone['polygon'],
                frame_resolution_wh=frame_wh
            )

            zone['PolygonZoneAnnotator'] = PolygonZoneAnnotator(
                zone=zone['PolygonZone'],
                color=sv.Color.WHITE,
                thickness=2,
                text_thickness=1,
                text_scale=0.3,
                text_padding=3
            )

            zones.append(zone)

    return zones


def setup_zone_below(detections, frame_wh):
    zones = []
    count_zone = 0
    for i in range(len(detections)):
        obj = detections[i]


        # choosing only baskets with confidence score >= 0.5
        if (obj.data.get('class_name')[0] == 'basket') & (obj.confidence[0] >= 0.5):
            count_zone += 1

            obj_xyxy = obj.xyxy[0]

            # get coordinates of 4 points of bounding box
            obj_p1 = np.array([obj_xyxy[0], obj_xyxy[1]])
            obj_p2 = np.array([obj_xyxy[0], obj_xyxy[3]])
            obj_p3 = np.array([obj_xyxy[2], obj_xyxy[3]])
            obj_p4 = np.array([obj_xyxy[2], obj_xyxy[1]])

            # get height and width of bounding box
            obj_height = obj_xyxy[3] - obj_xyxy[1]
            obj_width = obj_xyxy[2] - obj_xyxy[0]

            # polygon construction strategy
            # 1) polygon width will be width of "basket" bounding box increased on 10% (5% from left and right sides)
            # 2) polygon height will be height of "basket" bounding box
            # 3) polygon will be lower the basket bounding box on 0.5 height of it

            zone_p1 = [obj_p1[0] + 0.2 * obj_width, obj_p1[1] + 0.5 * obj_height]
            zone_p2 = [obj_p1[0] + 0.2 * obj_width, obj_p1[1] + 1.5 * obj_height]
            zone_p3 = [obj_p4[0] - 0.2 * obj_width, obj_p4[1] + 1.5 * obj_height]
            zone_p4 = [obj_p4[0] - 0.2 * obj_width, obj_p4[1] + 0.5 * obj_height]

            # round coordinates of each point of zone
            zone_p1 = list(map(round, zone_p1))
            zone_p2 = list(map(round, zone_p2))
            zone_p3 = list(map(round, zone_p3))
            zone_p4 = list(map(round, zone_p4))

            zone_coordinates = np.array([zone_p1, zone_p2, zone_p3, zone_p4])
            zone_name = f"Zone {count_zone}"

            zone = {
                'name': zone_name,
                'polygon': zone_coordinates,
                'count': 0
            }

            zone['PolygonZone'] = PolygonZone(
                polygon=zone['polygon'],
                frame_resolution_wh=frame_wh
            )

            zone['PolygonZoneAnnotator'] = PolygonZoneAnnotator(
                zone=zone['PolygonZone'],
                color=sv.Color.WHITE,
                thickness=2,
                text_thickness=1,
                text_scale=0.3,
                text_padding=3
            )

            zones.append(zone)

    return zones


def setup_zone_general(detections, frame_wh, zones_above, zones_below):
    general_zones = []

    for count_zone in range(len(zones_above)):

        y1 = zones_above[count_zone]['polygon'][0][1] # the upper y coordinate
        y2 = zones_below[count_zone]['polygon'][1][1] # the lower y coordinate
        width = zones_above[count_zone]['polygon'][3][0] - zones_above[count_zone]['polygon'][0][0]
        x1 = round(zones_above[count_zone]['polygon'][0][0] - 0.5 * width)
        x2 = round(zones_above[count_zone]['polygon'][3][0] + 0.5 * width)

        general_zone_p1 = [x1, y1]
        general_zone_p2 = [x1, y2]
        general_zone_p3 = [x2, y2]
        general_zone_p4 = [x2, y1]

        general_zone_coord = np.array([general_zone_p1, general_zone_p2, general_zone_p3, general_zone_p4])

        general_zone_name = f"Zone {count_zone + 1}"

        general_zone = {
            'name': general_zone_name,
            'polygon': general_zone_coord,
            'count': 0
        }

        general_zone['PolygonZone'] = PolygonZone(
            polygon=general_zone['polygon'],
            frame_resolution_wh=frame_wh
        )

        general_zone['PolygonZoneAnnotator'] = PolygonZoneAnnotator(
            zone=general_zone['PolygonZone'],
            color=sv.Color.WHITE,
            thickness=2,
            text_thickness=1,
            text_scale=0.3,
            text_padding=3
        )

        general_zones.append(general_zone)

    return general_zones


def choosing_largest_zone(zones):
    max_square = 0
    for zone in zones:
        zone_coord = zone.get('polygon')
        x1 = zone_coord[0][0]
        x2 = zone_coord[3][0]
        y1 = zone_coord[0][1]
        y2 = zone_coord[0][1]

        square = (x2 - x1) * (y2 - y1)

        if square > max_square:
            max_square = square

    for zone in zones:
        zone_coord = zone.get('polygon')
        x1 = zone_coord[0][0]
        x2 = zone_coord[3][0]
        y1 = zone_coord[0][1]
        y2 = zone_coord[0][1]

        if (x2 - x1) * (y2 - y1) == max_square:
            return zone
        

def video_callback(model, frame: np.ndarray, frame_wh, frame_num: int, zone_above, zone_below, zone_general, zone_above_triggered, zone_general_triggered, score, throws) -> np.ndarray:

    detections = detect(frame, model)

    # define zones by first frame
    if frame_num == 0:
        zones_above = setup_zone_above(detections, frame_wh)
        zones_below = setup_zone_below(detections, frame_wh)
        zones_general = setup_zone_general(detections, frame_wh, zones_above, zones_below)

        # choose only one zone above, below and general by greates size
        zone_above = choosing_largest_zone(zones_above)
        zone_below = choosing_largest_zone(zones_below)
        zone_general = choosing_largest_zone(zones_general)


    # bounding boxes labels
    labels = [
        f"{model.model.names[class_id]} {confidence:0.2f}"
        for confidence, class_id
        in zip(detections.confidence, detections.class_id)
    ]

    # draw bounding boxes with labels
    annotated_frame = frame.copy()
    annotated_frame = bounding_box_annotator.annotate(
        scene=annotated_frame,
        detections=detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels)

    # get person and ball bounding boxes coordinates of points of main diagonal
    person_p1 = person_p2 = (0, 0)
    ball_p1 = ball_p2 = (0, 0)

    # intersection between player and ball
    for obj in iter(detections):
        if obj[5].get('class_name') == 'person':
            person_x1, person_y1, person_x2, person_y2 = obj[0]
            person_p1 = (person_x1, person_y1)
            person_p2 = (person_x2, person_y2)

        if obj[5].get('class_name') == 'ball':
            ball_x1, ball_y1, ball_x2, ball_y2 = obj[0]
            ball_p1 = (ball_x1, ball_y1)
            ball_p2 = (ball_x2, ball_y2)

    # True, if person and ball bounding boxes intersect
    intersect_person_ball = rectangles_intersect(person_p1, person_p2, ball_p1, ball_p2)

    # Draw text for person and ball intersection
    text_anchor = sv.Point(x=500, y=200)
    if intersect_person_ball:
        annotated_frame = draw_text(scene=annotated_frame, text=f"Person and ball intersect: {intersect_person_ball}", text_anchor=text_anchor, text_scale=1, text_thickness=2, text_color=Color.GREEN)
    else:
        annotated_frame = draw_text(scene=annotated_frame, text=f"Person and ball intersect: {intersect_person_ball}", text_anchor=text_anchor, text_scale=1, text_thickness=2, text_color=Color.RED)

    zone_presence_above = zone_above['PolygonZone'].trigger(detections, 'ball')
    zone_presence_below = zone_below['PolygonZone'].trigger(detections, 'ball')
    zone_presence_general = zone_general['PolygonZone'].trigger(detections, 'ball')

    # True, if person and general zone bounding boxes intersect
    intersect_person_zone = rectangles_intersect(person_p1, person_p2, zone_general['polygon'][0], zone_general['polygon'][2])

    # write text for above zone
    text_anchor = sv.Point(x=500, y=50)
    if zone_presence_above and not intersect_person_zone and not intersect_person_ball:
        annotated_frame = draw_text(scene=annotated_frame, text=f"Ball in above zone: {zone_presence_above}", text_anchor=text_anchor, text_scale=1, text_thickness=2, text_color=Color.GREEN)
        zone_above['count'] += 1
        zone_above_triggered = True
    else:
        annotated_frame = draw_text(scene=annotated_frame, text=f"Ball in above zone: {zone_presence_above}", text_anchor=text_anchor, text_scale=1, text_thickness=2, text_color=Color.RED)

    # write text for below zone
    text_anchor = sv.Point(x=500, y=100)
    if zone_presence_below and not intersect_person_zone and not intersect_person_ball:
        annotated_frame = draw_text(scene=annotated_frame, text=f"Ball in below zone: {zone_presence_below}", text_anchor=text_anchor, text_scale=1, text_thickness=2, text_color=Color.GREEN)
        zone_below['count'] += 1
    else:
        annotated_frame = draw_text(scene=annotated_frame, text=f"Ball in below zone: {zone_presence_below}", text_anchor=text_anchor, text_scale=1, text_thickness=2, text_color=Color.RED)

    # write text for general zone
    text_anchor = sv.Point(x=500, y=150)
    if zone_presence_general and not intersect_person_zone and not intersect_person_ball:
        annotated_frame = draw_text(scene=annotated_frame, text=f"Ball in general zone: {zone_presence_general}", text_anchor=text_anchor, text_scale=1, text_thickness=2, text_color=Color.GREEN)
        zone_general['count'] += 1
        zone_general_triggered = True
    else:
        annotated_frame = draw_text(scene=annotated_frame, text=f"Ball in general zone: {zone_presence_general}", text_anchor=text_anchor, text_scale=1, text_thickness=2, text_color=Color.RED)



    # if the above zone is triggered by ball, we wait for the zone below to also be triggered by ball.
    # If the outcome when player touches the ball ccures earlier, than the zone below is triggered by ball, then we reset the zone_above_triggered to False.
    if zone_above_triggered and zone_presence_below and not intersect_person_zone and not intersect_person_ball:
        score += 1
        zone_above_triggered = False
    elif zone_above_triggered and intersect_person_ball and not intersect_person_zone:
        zone_above_triggered = False

    # if the general zone is triggered by ball, we wait till the player touches the ball. Then we conclude that the throw was made
    if zone_general_triggered and intersect_person_ball and not intersect_person_zone:
        throws += 1
        zone_general_triggered = False



    annotated_frame = zone_above['PolygonZoneAnnotator'].annotate(
        scene=annotated_frame,
        label=f"{zone_above['name']}: {zone_above['count']}"
    )

    annotated_frame = zone_below['PolygonZoneAnnotator'].annotate(
        scene=annotated_frame,
        label=f"{zone_below['name']}: {zone_below['count']}"
    )

    annotated_frame = zone_general['PolygonZoneAnnotator'].annotate(
        scene=annotated_frame,
        label=f"{zone_general['name']}: {zone_general['count']}"
    )

    # draw text for score
    text_anchor = sv.Point(x=130, y=50)
    annotated_frame = draw_text(scene=annotated_frame, text=f"Score: {score}", text_anchor=text_anchor, text_scale=1.5, text_thickness=2)

    # draw text for throws
    text_anchor = sv.Point(x=130, y=100)
    annotated_frame = draw_text(scene=annotated_frame, text=f"Throws: {throws}", text_anchor=text_anchor, text_scale=1.5, text_thickness=2)


    return annotated_frame, zone_above, zone_below, zone_general, zone_above_triggered, zone_general_triggered, score, throws


import asyncio
from src.process_video.connection_manager import ConnectionManager

def video_process(SOURCE_VIDEO_PATH, TARGET_VIDEO_PATH, manager: ConnectionManager, loop, result_holder, stop_event):
    HOME = os.getcwd()

    MODEL = f"{HOME}/best.pt"
    model = YOLO(MODEL)

    frames_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)
    video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)
    total_frames = video_info.total_frames

    zone_above = {}
    zone_below = {}
    zone_general = {}
    zone_above_triggered = False
    zone_general_triggered = False
    score = 0
    throws = 0

    with sv.VideoSink(target_path=TARGET_VIDEO_PATH, video_info=video_info) as sink:
        for i, frame in enumerate(frames_generator):
            if stop_event.is_set():
                print("Stopping video processing as requested.")
                break

            # Infer
            annotated_frame, zone_above, zone_below, zone_general, zone_above_triggered, zone_general_triggered, score, throws = \
            video_callback(model, frame, video_info.resolution_wh, i, zone_above, zone_below, zone_general, zone_above_triggered, zone_general_triggered, score, throws)

            print(f"Processing frame {i+1}/{total_frames}, Score {score}, Throws {throws}")
            sink.write_frame(frame=annotated_frame)

            progress = (i+1) / total_frames * 100
            asyncio.run_coroutine_threadsafe(manager.send_progress(progress), loop)
            # asyncio.run_coroutine_threadsafe(manager.send_progress(f"{progress}"), loop)

    misses = throws - score

    # Store results in the result holder
    result_holder['score'] = score
    result_holder['misses'] = misses
    result_holder['throws'] = throws

    if not stop_event.is_set():
        asyncio.run_coroutine_threadsafe(manager.send_completion_message(), loop)


    

    # return score, misses, throws




# def progress_bar_decorator(func):
#     def wrapper(*args, **kwargs):
#         source_video_path = args[0]  # Assuming the first argument is SOURCE_VIDEO_PATH
#         video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
#         total_frames = video_info.total_frames

#         with tqdm(total=total_frames, desc="Processing video frames") as pbar:
#             def callback_wrapper(*cb_args, **cb_kwargs):
#                 result = video_callback(*cb_args, **cb_kwargs)
#                 pbar.update(1)
#                 return result

#             return func(*args, video_callback=callback_wrapper, **kwargs)
#     return wrapper

# @progress_bar_decorator
# def video_process(SOURCE_VIDEO_PATH, TARGET_VIDEO_PATH, video_callback):
#     HOME = os.getcwd()
#     MODEL = f"{HOME}/best.pt"
#     model = YOLO(MODEL)

#     frames_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)
#     video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)

#     zone_above = {}
#     zone_below = {}
#     zone_general = {}
#     zone_above_triggered = False
#     zone_general_triggered = False
#     score = 0
#     throws = 0

#     with sv.VideoSink(target_path=TARGET_VIDEO_PATH, video_info=video_info) as sink:
#         for i, frame in enumerate(frames_generator):
#             # Infer
#             annotated_frame, zone_above, zone_below, zone_general, zone_above_triggered, zone_general_triggered, score, throws = \
#                 video_callback(model, frame, video_info.resolution_wh, i, zone_above, zone_below, zone_general, zone_above_triggered, zone_general_triggered, score, throws)

#             print(f"Processing frame {i}, Score {score}, Throws {throws}")
#             sink.write_frame(frame=annotated_frame)

#     misses = throws - score
#     return score, misses, throws