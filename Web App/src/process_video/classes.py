import numpy as np
import numpy.typing as npt
from typing import Iterable, Optional, Tuple
import cv2
from supervision import Detections
from supervision.detection.utils import clip_boxes, polygon_to_mask
from supervision.draw.color import Color
from supervision.draw.utils import draw_polygon, draw_text
from supervision.geometry.core import Position
from supervision.geometry.utils import get_polygon_center
from supervision.utils.internal import deprecated_parameter
import supervision as sv

def point_in_rect(px: int, py: int, rx1: int, ry1: int, rx2: int, ry2: int) -> bool:
    """
    Check if a point (px, py) is inside a rectangle defined by its top-left (rx1, ry1)
    and bottom-right (rx2, ry2) coordinates.

    Parameters:
    px (float): x-coordinate of the point.
    py (float): y-coordinate of the point.
    rx1 (float): x-coordinate of the rectangle's top-left corner.
    ry1 (float): y-coordinate of the rectangle's top-left corner.
    rx2 (float): x-coordinate of the rectangle's bottom-right corner.
    ry2 (float): y-coordinate of the rectangle's bottom-right corner.

    Returns:
    bool: True if the point is inside the rectangle, False otherwise.
    """
    return rx1 <= px <= rx2 and ry1 <= py <= ry2

def rectangles_intersect(r1_p1, r1_p2,
                         r2_p1, r2_p2) -> bool:
    """
    Check if two rectangles intersect. Each rectangle is defined by two points:
    the top-left and the bottom-right corners.

    Parameters:
    r1_p1 (tuple[float, float]): Top-left corner of the first rectangle (x1, y1).
    r1_p2 (tuple[float, float]): Bottom-right corner of the first rectangle (x2, y2).
    r2_p1 (tuple[float, float]): Top-left corner of the second rectangle (x1, y1).
    r2_p2 (tuple[float, float]): Bottom-right corner of the second rectangle (x2, y2).

    Returns:
    bool: True if the rectangles intersect, False otherwise.
    """
    # Unpack rectangle points
    r1_x1, r1_y1 = r1_p1
    r1_x2, r1_y2 = r1_p2
    r2_x1, r2_y1 = r2_p1
    r2_x2, r2_y2 = r2_p2

    # Define the vertices for both rectangles
    r1_vertices = [(r1_x1, r1_y1), (r1_x2, r1_y1), (r1_x2, r1_y2), (r1_x1, r1_y2)]
    r2_vertices = [(r2_x1, r2_y1), (r2_x2, r2_y1), (r2_x2, r2_y2), (r2_x1, r2_y2)]

    # Check if any vertex of rectangle 1 is inside rectangle 2 or vice versa
    return any(point_in_rect(px, py, r2_x1, r2_y1, r2_x2, r2_y2) for px, py in r1_vertices) or \
           any(point_in_rect(px, py, r1_x1, r1_y1, r1_x2, r1_y2) for px, py in r2_vertices)

# define annotation classes
bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.4)

class PolygonZone:
    """
    A class for defining a polygon-shaped zone within a frame for detecting objects.

    Attributes:
        polygon (np.ndarray): A polygon represented by a numpy array of shape
            `(N, 2)`, containing the `x`, `y` coordinates of the points.
        frame_resolution_wh (Tuple[int, int]): The frame resolution (width, height)
        triggering_anchors (Iterable[sv.Position]): A list of positions specifying
            which anchors of the detections bounding box to consider when deciding on
            whether the detection fits within the PolygonZone
            (default: (sv.Position.BOTTOM_CENTER,)).
        current_count (int): The curren|t count of detected objects within the zone
        mask (np.ndarray): The 2D bool mask for the polygon zone
    """

    @deprecated_parameter(
        old_parameter="triggering_position",
        new_parameter="triggering_anchors",
        map_function=lambda x: [x],
        warning_message="`{old_parameter}` in `{function_name}` is deprecated and will "
        "be remove in `supervision-0.23.0`. Use '{new_parameter}' "
        "instead.",
    )
    def __init__(
        self,
        polygon: npt.NDArray[np.int64],
        frame_resolution_wh: Tuple[int, int],
        triggering_anchors: Iterable[Position] = (Position.BOTTOM_CENTER,),

    ):
        self.polygon = polygon.astype(int)
        self.frame_resolution_wh = frame_resolution_wh
        self.triggering_anchors = triggering_anchors

        self.current_count = 0

        width, height = frame_resolution_wh
        self.mask = polygon_to_mask(
            polygon=polygon, resolution_wh=(width + 1, height + 1)
        )

    def trigger(self, detections: Detections, class_id):
        """
        Determines if the spicific objects are within the polygon zone.

        Return:
        False, if none of the ojects of the specified class are within the polygon zone.
        True, if at least one of the ojects of the specified class are within the polygon zone.
        """

        triggers = []

        for obj in iter(detections):
            if obj[3] == class_id: # check for ball
                obj_x1, obj_y1, obj_x2, obj_y2 = obj[0]
                obj_p1 = (obj_x1, obj_y1)
                obj_p2 = (obj_x2, obj_y2)

                zone_p1 = self.polygon[0]
                zone_p2 = self.polygon[2]

                triggers.append(rectangles_intersect(obj_p1, obj_p2, zone_p1, zone_p2))

        return any(triggers)


class PolygonZoneAnnotator:
    """
    A class for annotating a polygon-shaped zone within a
        frame with a count of detected objects.

    Attributes:
        zone (PolygonZone): The polygon zone to be annotated
        color (Color): The color to draw the polygon lines
        thickness (int): The thickness of the polygon lines, default is 2
        text_color (Color): The color of the text on the polygon, default is black
        text_scale (float): The scale of the text on the polygon, default is 0.5
        text_thickness (int): The thickness of the text on the polygon, default is 1
        text_padding (int): The padding around the text on the polygon, default is 10
        font (int): The font type for the text on the polygon,
            default is cv2.FONT_HERSHEY_SIMPLEX
        center (Tuple[int, int]): The center of the polygon for text placement
        display_in_zone_count (bool): Show the label of the zone or not. Default is True
    """

    def __init__(
        self,
        zone: PolygonZone,
        color: Color,
        thickness: int = 2,
        text_color: Color = Color.BLACK,
        text_scale: float = 0.5,
        text_thickness: int = 1,
        text_padding: int = 10,
        display_in_zone_count: bool = True,
    ):
        self.zone = zone
        self.color = color
        self.thickness = thickness
        self.text_color = text_color
        self.text_scale = text_scale
        self.text_thickness = text_thickness
        self.text_padding = text_padding
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.center = get_polygon_center(polygon=zone.polygon)
        self.display_in_zone_count = display_in_zone_count

    def annotate(self, scene: np.ndarray, label: Optional[str] = None) -> np.ndarray:
        """
        Annotates the polygon zone within a frame with a count of detected objects.

        Parameters:
            scene (np.ndarray): The image on which the polygon zone will be annotated
            label (Optional[str]): An optional label for the count of detected objects
                within the polygon zone (default: None)

        Returns:
            np.ndarray: The image with the polygon zone and count of detected objects
        """
        annotated_frame = draw_polygon(
            scene=scene,
            polygon=self.zone.polygon,
            color=self.color,
            thickness=self.thickness,
        )

        if self.display_in_zone_count:
            annotated_frame = draw_text(
                scene=annotated_frame,
                text=str(self.zone.current_count) if label is None else label,
                text_anchor=self.center,
                background_color=self.color,
                text_color=self.text_color,
                text_scale=self.text_scale,
                text_thickness=self.text_thickness,
                text_padding=self.text_padding,
                text_font=self.font,
            )

        return annotated_frame