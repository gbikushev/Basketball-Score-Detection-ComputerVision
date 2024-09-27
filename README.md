# Basketball Throw and Goal Detection Web Application

This project is a web-based platform that uses a custom-trained YOLOv8 deep learning model and a unique algorithm to automatically detect basketball throws and classify successful shots from videos recorded during training sessions. The application allows players to analyze their shooting performance by providing detailed statistical insights, including the total number of throws, successful shots, and interactive visualizations showcasing shooting accuracy trends over time.

## Table of Contents
- [Project Overview](#project-overview)
- [YOLO Model](#yolo-model)
  - [Dataset creation](#dataset-creation)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
  - [Model Inference](#model-inference)
- [Throws and Score Detection Algorithm](#throws-and-score-detection-algorithm)
  - [The Logic of the Algorithm](#the-logic-of-the-algorithm)
  - [Testing Dataset](#testing-dataset)
  - [Algorithm Evaluation](#algorithm-evaluation)
  - [Algorithm Evaluation Conclusion](#algorithm-evaluation-conclusion)
  - [Video Requirements](#video-requirements)
- [Web App](#web-app)
  - [Main Technologies](#main-technologies)
  - [Functionality + Screenshots](#functionality--screenshots)
  - [Database](#database)
  - [API Documentation](#api-documentation)
  - [Authentication In Web Application](#authentication-in-web-application)
- [Conclusion](#conclusion)


## Project Overview

This project automates the detection of basketball throws and successful shots using a custom-trained YOLOv8 model. By analyzing uploaded videos from basketball training sessions, the web application provides a detailed statistical breakdown of shooting performance, including goals and misses. The core features include video processing, shot classification (whether it's successful or not), and insightful interactive visualizations.

For processing by the algorithm, it is required to upload videos with the following types of throws:
1) Free Throws
2) 3-Point Throws

# YOLO Model
## Dataset creation
I collected 14 YouTube Shorts videos (camera installed on some height) and 14 videos from external participants (camera located on the ground), with both indoor and outdoor environments. 

| Video Num | Video Category       | Type of Action                        | Camera Location | Frames Extracted | Ball Color            | Environment            |
|-----------|----------------------|---------------------------------------|-----------------|------------------|-----------------------|------------------------|
| [1](https://www.youtube.com/shorts/PNWtLkEbjUo)  | YouTube Shorts       | Dribbling, Layups                    | Height          | 205              | Orange                | Indoors                |
| [2](https://www.youtube.com/shorts/SAVKpoH2YPs)  | YouTube Shorts       | Dunks                                 | Height          | 66               | Orange                | Indoors                |
| [3](https://www.youtube.com/shorts/3UEitmXTNAg)  | YouTube Shorts       | Layups                                | Height          | 249              | Orange                | Indoors                |
| [4](https://www.youtube.com/shorts/5PXdig269Q8)  | YouTube Shorts       | Layups                                | Height          | 111              | Orange                | Indoors                |
| [5](https://www.youtube.com/shorts/gPkBpZhvDyE)  | YouTube Shorts       | Dribbling, Throws                     | Height          | 110              | Orange                | Indoors                |
| [6](https://www.youtube.com/shorts/rSBqoHGfztM)  | YouTube Shorts       | Throws, Layups                        | Height          | 298              | White red blue        | Outdoors               |
| [7](https://www.youtube.com/shorts/G1WawtWaGzA)  | YouTube Shorts       | Layups                                | Height          | 73               | Black                 | Outdoors               |
| [8](https://www.youtube.com/shorts/lLyIqHKH7nw)  | YouTube Shorts       | Dribbling, Throws                     | Height          | 171              | Orange                | Outdoors, Indoors      |
| [9](https://www.youtube.com/shorts/FeprGhJlNvw)  | YouTube Shorts       | Throws                                | Height          | 185              | Orange                | Outdoors, Indoors      |
| [10](https://www.youtube.com/shorts/bOVbkELNbmI) | YouTube Shorts       | Throws                                | Height          | 129              | Orange                | Outdoors               |
| [11](https://www.youtube.com/shorts/w1YRG1UhgPY) | YouTube Shorts       | Dribbling, Layups                     | Height          | 83               | Orange                | Outdoors               |
| [12](https://www.youtube.com/shorts/CLPT8LjQ8rc) | YouTube Shorts       | Dribbling, Throws, Dunks              | Height          | 204              | Orange, Red blue      | Outdoors               |
| [13](https://www.youtube.com/shorts/l8MyPg5Xk2c) | YouTube Shorts       | Dribbling, Throws, Dunks              | Height          | 377              | Orange                | Outdoors               |
| [14](https://www.youtube.com/shorts/5sUhFYATYeM) | YouTube Shorts       | Dribbling, Throws                     | Height          | 106              | Orange                | Indoors                |
| 15         | External Contributor | Dribbling, Throws                     | Ground          | 154              | Orange                | Indoors                |
| 16         | External Contributor | Throws                                | Ground          | 68               | Orange                | Indoors                |
| 17         | External Contributor | Dribbling, Throws, Layups             | Ground          | 123              | Orange                | Indoors                |
| 18         | External Contributor | Throws                                | Ground          | 124              | Orange                | Indoors                |
| 19         | External Contributor | Dribbling, Throws, Layups             | Ground          | 232              | Orange                | Indoors                |
| 20         | External Contributor | Dribbling, Throws, Layups             | Ground          | 168              | Orange                | Indoors                |
| 21         | External Contributor | Dribbling, Throws, Layups             | Ground          | 189              | Orange                | Indoors                |
| 22         | External Contributor | Dribbling, Throws, Layups             | Ground          | 239              | Orange                | Indoors                |
| 23         | External Contributor | Dribbling, Throws, Layups             | Ground          | 244              | Orange                | Indoors                |
| 24         | External Contributor | Dribbling, Throws, Layups             | Ground          | 245              | Orange                | Indoors                |
| 25         | External Contributor | Dribbling, Throws, Layups             | Ground          | 236              | Orange                | Indoors                |
| 26         | External Contributor | Dribbling, Throws, Layups             | Ground          | 195              | Orange                | Indoors                |
| 27         | External Contributor | Throws                                | Ground          | 157              | Orange                | Indoors                |
| 28         | External Contributor | Throws                                | Ground          | 159              | Orange                | Indoors                |
| **Total**  | -                    | -                                     | -               | **4900**          | -                     | -                      |



Videos were cut into frames and manually annotated using classes «person», «ball», «basket». Total number of frames is 4,900 with 16,103 object instances.

<p align="center">
  <img src="pics_readme/data_collect_1.jpg" width="600" title="Counts of Instances by Class and Subset">
</p>
<p align="center"><em>Counts of Instances by Class and Subset</em></p>

<p align="center">
  <img src="pics_readme/data_collect_2.jpg" width="600" title="Counts of Objects by class in all the frames">
</p>
<p align="center"><em>Counts of Objects by class in all the frames</em></p>

After preprocessing steps:
- **Training set** (80%): 6017 images
- **Validation set** (13%): 981 images
- **Testing set** (7%): 488 images


## Model Training
The model was pre-trained on the COCO dataset and retrained on the custom-labeled dataset for 100 epochs.

<p align="center">
  <img src="pics_readme/training_1.png" width="700" title="Training results">
</p>
<p align="center"><em>Training results</em></p>

## Model Evaluation
After evaluating the model objects detecting capabilities on the testing set, we have the results in table
and pictures of confusion matrixes. The table demonstrates high values close to 1 for such evaluation 
metrics as Recall, Precision and mean average precision for both IoU thresholds. The confusion 
matrixes also show a large proportion of true positives for all class.

<p align="center">
  <img src="pics_readme/confusion_matrices.png" width="700" title="Object Detection Performance: Confusion Matrices Overview">
</p>
<p align="center"><em>Object Detection Performance: Confusion Matrices Overview</em></p>


**Model evaluation on testing set:**

<div align="center">

| Class  | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|--------|--------|-----------|-----------|--------|-------|----------|
| all    | 488    | 1612      | 0.989     | 0.982  | 0.993 | 0.878    |
| ball   | 488    | 431       | 0.983     | 0.968  | 0.991 | 0.807    |
| basket | 488    | 550       | 0.994     | 0.991  | 0.994 | 0.934    |
| person | 488    | 631       | 0.989     | 0.989  | 0.995 | 0.893    |

</div>

## Model Inference
Picture demonstrates my YOLO model objects detection capabilities on completely unseen data. As can be observed,
in this case it perfectly detects actual objects using bounding boxes on the provided frames: person, basket and ball.

<p align="center">
  <img src="pics_readme/unseen_data.png" width="900" title="Model inference on unseen data">
</p>
<p align="center"><em>Model inference on unseen data</em></p>


# Throws and Score Detection Algorithm
## The Logic of the Algorithm
### Building Rectangular zones
The logic of the algorithm for detecting throws and successful shots is based on obtaining and managing the coordinates of bounding boxes of the player, the ball, and the basket from each frame, as the output of our custom YOLOv8 model, and detecting the "ball" object in 3 certain rectangular zones built next to the basket.

For building these zones, I used PolygonZone from "supervision" Python library, a class for defining
a polygon-shaped zone within a frame for detecting objects. Each of these 3 rectangular zones performs
its own function and is constructed as follows:

**For score detection:**
1. Zone above the basket (zone_above)
    - Zone width: the width of the "basket" bounding box increased on 10% (5% from both left and right sides).
    - Zone height: 60% of the height of the "basket" bounding box
    - Zone location: higher on 10% of the "basket" bounding box height from the upper bound of "basket" bounding box.
2. Zone below the basket (zone_below)
    - Zone width: the width of the "basket" bounding box decreased on 95% (47.5% from both left and right sides).
    - Zone height: 80% of the height of the "basket" bounding box.
    - Zone location: lower on 70% of the "basket" bounding box height from the lower bound of "basket" bounding box.

**For throw detection:**
1. More space coverage zone (zone_general)
    - Zone width: the width of the zone_above increased on 100% (50% from both left and right sides).
    - Zone height: 260% of the height of the zone_above.
    - Zone location: lower on 10% of the height of zone_above from the lower bound of zone_above.

Figure showcases how these zones look like at the frames with all other detected objects presented:

<p align="center">
  <img src="pics_readme/zones_example.png" width="900" title="Examples of building zone_above, zone_below, zone_general">
</p>
<p align="center"><em>Examples of building zone_above, zone_below, zone_general</em></p>

### Algorithm Pipeline
The schema below represents how each frame is being processed by our algorithm:

<p align="center">
  <img src="pics_readme/final_schema.png" width="900" title="Pipeline diagram of how every frame is being processed by the algorithm">
</p>
<p align="center"><em>Pipeline diagram of how every frame is being processed by the algorithm</em></p>

After the video was processed by the proposed framework, the resulting video is annotated with indicators that reflect the ball’s presence in the certain zone, as well as intersection between person and ball. The upper left corner shows the values for number of throws and successful shots. Finally, number of misses can be calculated as:

$$ Misses = Throws - Score $$

Random frames from videos processed by the algorithm:
<p align="center">
  <img src="pics_readme/alg_example.png" width="900" title="Random frames from videos processed by the algorithm">
</p>
<p align="center"><em>Random frames from videos processed by the algorithm</em></p>



##  Testing Dataset
I myself have recorded and prepared new 16 videos especially for algorithm evaluation,
previously unseen by my YOLO model during its training and validation. These videos are 1-2 minute
long, recorded both indoors and outdoors from both required camera heights (on the ground and with
the tripod at a height of 1.5 meters) and required camera positions (red dots on figure 11) of an athlete
throwing the ball into the basketball hoop from the free throw line and from three-point range, which
the model has not seen before during its training and validation processes. 

These videos were recorded with iPhone 15 Pro with resolution 1080p (Full HD) and 60 fps. The recorded videos are shown in table below:

<div align="center">

| Video Number | Court Location | Camera Height | Camera Position                   | Throw Type        | Video Duration | Number of Throws | Number of Goals | Number of Misses |
|--------------|----------------|---------------|------------------------------------|-------------------|----------------|------------------|-----------------|------------------|
| [1](https://drive.google.com/file/d/1OFHD16w_Bo5I09hUECIOpeh5sW-Oj24L/view?usp=sharing)  | Outdoors       | Ground        | Right Side From the Hoop           | Free Throws       | 1:54           | 16               | 10              | 6                |
| [2](https://drive.google.com/file/d/1LPx-EUCUzJvdhgz1pvO8NoKhlHGNtBBQ/view?usp=sharing)  | Outdoors       | Ground        | Right Side From the Hoop           | 3-Point Throws    | 1:39           | 12               | 4               | 8                |
| [3](https://drive.google.com/file/d/1rGLxsvm6vgzKW6xjYs4_WmxlCWnFkeOb/view?usp=sharing)  | Outdoors       | Ground        | Left Side From the Hoop            | Free Throws       | 1:41           | 12               | 9               | 3                |
| [4](https://drive.google.com/file/d/181MFpt5GEM8CLL4ShfrWj_GYfYIwuEXR/view?usp=sharing)  | Outdoors       | Ground        | Left Side From the Hoop            | 3-Point Throws    | 1:43           | 13               | 1               | 12               |
| [5](https://drive.google.com/file/d/18murzjzOefqR2cA4t7Dhxhk6ODgoq4fM/view?usp=sharing)  | Outdoors       | Tripod        | Right Side From the Hoop           | Free Throws       | 1:30           | 13               | 8               | 5                |
| [6](https://drive.google.com/file/d/1J4Qjk_j713CAuK6LvtzRI702N73m1N0Q/view?usp=sharing)  | Outdoors       | Tripod        | Right Side From the Hoop           | 3-Point Throws    | 2:02           | 11               | 4               | 7                |
| [7](https://drive.google.com/file/d/1GN-UivOLhH6XYPImGSdJOirQUsrLU92L/view?usp=sharing)  | Outdoors       | Tripod        | Left Side From the Hoop            | Free Throws       | 1:48           | 16               | 6               | 11               |
| [8](https://drive.google.com/file/d/12QTMcgPaDcG1DJk6UEgMPxvIU7tSj25-/view?usp=sharing)  | Outdoors       | Tripod        | Left Side From the Hoop            | 3-Point Throws    | 1:22           | 8                | 2               | 6                |
| [9](https://drive.google.com/file/d/1uXZcHD0n2_5lTRId14maGQEFICMuRFEJ/view?usp=sharing)  | Indoors        | Ground        | Right Side From the Hoop           | Free Throws       | 1:56           | 15               | 7               | 8                |
| [10](https://drive.google.com/file/d/1_RW9FimDBAE9_qoK4g_MijlL6s82Dfyx/view?usp=sharing) | Indoors        | Ground        | Right Side From the Hoop           | 3-Point Throws    | 1:19           | 10               | 5               | 5                |
| [11](https://drive.google.com/file/d/1oI9BGjrLtI731tb-t1O2IMkvClJlPNpN/view?usp=sharing) | Indoors        | Ground        | Left Side From the Hoop            | Free Throws       | 1:58           | 14               | 4               | 10               |
| [12](https://drive.google.com/file/d/1xwJdUvDbDGR9T3RoAzUBgpEWVmZrwBLl/view?usp=sharing) | Indoors        | Ground        | Left Side From the Hoop            | 3-Point Throws    | 1:00           | 7                | 1               | 6                |
| [13](https://drive.google.com/file/d/1k07UnfbxTJ9Cg7oRHmHGdv8ic1X1lWA7/view?usp=sharing) | Indoors        | Tripod        | Right Side From the Hoop           | Free Throws       | 1:32           | 19               | 10              | 9                |
| [14](https://drive.google.com/file/d/1KvUGpuA5RqBN1AsxbfxVpaqvN_f3WvyQ/view?usp=sharing) | Indoors        | Tripod        | Right Side From the Hoop           | 3-Point Throws    | 1:43           | 13               | 5               | 8                |
| [15](https://drive.google.com/file/d/1GF7zUVn6m3_4onBUrpg2vh6FkFbec6Tm/view?usp=sharing) | Indoors        | Tripod        | Left Side From the Hoop            | Free Throws       | 1:55           | 16               | 8               | 8                |
| [16](https://drive.google.com/file/d/1UsLflhTKPdCTrvPpCiPl4VPma_o_ORvs/view?usp=sharing) | Indoors        | Tripod        | Left Side From the Hoop            | 3-Point Throws    | 1:00           | 7                | 2               | 5                |

</div>

## Algorithm Evaluation

### Actual vs Algorithm-predicted results of shooting accuracy

<div align="center">

| Video Number | Court Location | Camera Height | Camera Position                   | Throws Type      | Predicted Throws | Predicted Score | Actual Throws | Actual Score | Difference Throws | Difference Score |
|--------------|----------------|---------------|------------------------------------|------------------|------------------|-----------------|---------------|--------------|-------------------|------------------|
| [1](https://drive.google.com/file/d/1kv1U0XfDP2L4RrdXN8_xql9yDS44q9dT/view?usp=sharing)  | Outdoors       | Ground        | Right Side From the Hoop           | Free Throws       | 16               | 10              | 16            | 10           | 0                 | 0                |
| [2](https://drive.google.com/file/d/17snBxQnujO61sUlmxKxlNgP1rBCj2h16/view?usp=sharing)  | Outdoors       | Ground        | Right Side From the Hoop           | 3-Point Throws    | 12               | 6               | 12            | 4            | 0                 | 2                |
| [3](https://drive.google.com/file/d/1--sU_ijpy_ruj6c9xkjmVtpZc3Nf_xCr/view?usp=sharing)  | Outdoors       | Ground        | Left Side From the Hoop            | Free Throws       | 14               | 5               | 12            | 9            | 2                 | 4                |
| [4](https://drive.google.com/file/d/1-1qOl0TAYmaw8fZSJS7qzXbEC7vYjxS-/view?usp=sharing)  | Outdoors       | Ground        | Left Side From the Hoop            | 3-Point Throws    | 16               | 1               | 13            | 1            | 3                 | 0                |
| [5](https://drive.google.com/file/d/1-5lopGm0VivIYd3ZxRB9b3iTuSeLmMOe/view?usp=sharing)  | Outdoors       | Tripod        | Right Side From the Hoop           | Free Throws       | 15               | 4               | 13            | 8            | 2                 | 4                |
| [6](https://drive.google.com/file/d/1-7Bozphm-I9wkeZwVL2TKSZKOdKn5MI9/view?usp=sharing)  | Outdoors       | Tripod        | Right Side From the Hoop           | 3-Point Throws    | 11               | 3               | 11            | 4            | 0                 | 1                |
| [7](https://drive.google.com/file/d/1-6S_47b1Vgyi4RXn6Oo3qOx7aKrziSPC/view?usp=sharing)  | Outdoors       | Tripod        | Left Side From the Hoop            | Free Throws       | 16               | 4               | 16            | 6            | 0                 | 2                |
| [8](https://drive.google.com/file/d/1-G_Y0n57JzOgMjnLtrwAT20D4-blU4yq/view?usp=sharing)  | Outdoors       | Tripod        | Left Side From the Hoop            | 3-Point Throws    | 8                | 3               | 8             | 2            | 0                 | 1                |
| [9](https://drive.google.com/file/d/1-VjUzP2sdrspId_kutI_PjUdh781x8fA/view?usp=sharing)  | Indoors        | Ground        | Right Side From the Hoop           | Free Throws       | 16               | 5               | 15            | 7            | 1                 | 2                |
| [10](https://drive.google.com/file/d/1-aj94bD3daB13TeeqBy6nuGf7s-ORGJf/view?usp=sharing) | Indoors        | Ground        | Right Side From the Hoop           | 3-Point Throws    | 10               | 6               | 10            | 5            | 0                 | 1                |
| [11](https://drive.google.com/file/d/1-P1X2uNg3IS9zC0p0alpWN_St3a05O67/view?usp=sharing) | Indoors        | Ground        | Left Side From the Hoop            | Free Throws       | 14               | 5               | 14            | 4            | 0                 | 1                |
| [12](https://drive.google.com/file/d/1-UmCmBwb-0FpceBTcsHYTaCV9dd7XRMT/view?usp=sharing) | Indoors        | Ground        | Left Side From the Hoop            | 3-Point Throws    | 7                | 2               | 7             | 1            | 0                 | 1                |
| [13](https://drive.google.com/file/d/1HBb4dpWTUzFEfrwDjm2L80TnYY4_F5t2/view?usp=sharing) | Indoors        | Tripod        | Right Side From the Hoop           | Free Throws       | 19               | 11              | 19            | 10           | 0                 | 1                |
| [14](https://drive.google.com/file/d/1-4hqrJqYpzQL95w_gxMCVZy2Dc6NZI9e/view?usp=sharing) | Indoors        | Tripod        | Right Side From the Hoop           | 3-Point Throws    | 13               | 6               | 13            | 5            | 0                 | 1                |
| [15](https://drive.google.com/file/d/1-u_fX8oM6WtzA1Hdb3LJiKRwcIyLXvur/view?usp=sharing) | Indoors        | Tripod        | Left Side From the Hoop            | Free Throws       | 16               | 9               | 16            | 8            | 0                 | 1                |
| [16](https://drive.google.com/file/d/1-0x2RXGCpw9eT5teFGixug3kUCWY6NAK/view?usp=sharing) | Indoors        | Tripod        | Left Side From the Hoop            | 3-Point Throws    | 7                | 2               | 7             | 2            | 0                 | 0                |

</div>

### Overview of throws identification results
The table below provides a detailed analysis of the algorithm’s performance in identifying basketball throws, focusing on two key metrics: Identification Accuracy and Identification Precision:

<div align="center">

| Video Num                                                                                                     | Actual Throws | Predicted Throws | Identified Correctly | Identified Incorrectly | Identification Accuracy | Identification Precision |
|---------------------------------------------------------------------------------------------------------------|---------------|------------------|----------------------|------------------------|--------------------------|---------------------------|
| [1](https://drive.google.com/file/d/1kv1U0XfDP2L4RrdXN8_xql9yDS44q9dT/view?usp=sharing)                        | 16            | 16               | 16                   | 0                      | 1.0000                   | 1.0000                    |
| [2](https://drive.google.com/file/d/17snBxQnujO61sUlmxKxlNgP1rBCj2h16/view?usp=sharing)                        | 12            | 12               | 12                   | 0                      | 1.0000                   | 1.0000                    |
| [3](https://drive.google.com/file/d/1--sU_ijpy_ruj6c9xkjmVtpZc3Nf_xCr/view?usp=sharing)                        | 12            | 14               | 12                   | 2                      | 1.0000                   | 0.8571                    |
| [4](https://drive.google.com/file/d/1-1qOl0TAYmaw8fZSJS7qzXbEC7vYjxS-/view?usp=sharing)                        | 13            | 16               | 13                   | 3                      | 1.0000                   | 0.8125                    |
| [5](https://drive.google.com/file/d/1-5lopGm0VivIYd3ZxRB9b3iTuSeLmMOe/view?usp=sharing)                        | 13            | 15               | 13                   | 2                      | 1.0000                   | 0.8667                    |
| [6](https://drive.google.com/file/d/1-7Bozphm-I9wkeZwVL2TKSZKOdKn5MI9/view?usp=sharing)                        | 11            | 11               | 11                   | 0                      | 1.0000                   | 1.0000                    |
| [7](https://drive.google.com/file/d/1-6S_47b1Vgyi4RXn6Oo3qOx7aKrziSPC/view?usp=sharing)                        | 16            | 16               | 16                   | 0                      | 1.0000                   | 1.0000                    |
| [8](https://drive.google.com/file/d/1-G_Y0n57JzOgMjnLtrwAT20D4-blU4yq/view?usp=sharing)                        | 8             | 8                | 8                    | 0                      | 1.0000                   | 1.0000                    |
| [9](https://drive.google.com/file/d/1-VjUzP2sdrspId_kutI_PjUdh781x8fA/view?usp=sharing)                        | 15            | 16               | 15                   | 1                      | 1.0000                   | 0.9375                    |
| [10](https://drive.google.com/file/d/1-aj94bD3daB13TeeqBy6nuGf7s-ORGJf/view?usp=sharing)                       | 10            | 10               | 10                   | 0                      | 1.0000                   | 1.0000                    |
| [11](https://drive.google.com/file/d/1-P1X2uNg3IS9zC0p0alpWN_St3a05O67/view?usp=sharing)                       | 14            | 14               | 14                   | 0                      | 1.0000                   | 1.0000                    |
| [12](https://drive.google.com/file/d/1-UmCmBwb-0FpceBTcsHYTaCV9dd7XRMT/view?usp=sharing)                       | 7             | 7                | 7                    | 0                      | 1.0000                   | 1.0000                    |
| [13](https://drive.google.com/file/d/1HBb4dpWTUzFEfrwDjm2L80TnYY4_F5t2/view?usp=sharing)                       | 19            | 19               | 19                   | 0                      | 1.0000                   | 1.0000                    |
| [14](https://drive.google.com/file/d/1-4hqrJqYpzQL95w_gxMCVZy2Dc6NZI9e/view?usp=sharing)                       | 13            | 13               | 13                   | 0                      | 1.0000                   | 1.0000                    |
| [15](https://drive.google.com/file/d/1-u_fX8oM6WtzA1Hdb3LJiKRwcIyLXvur/view?usp=sharing)                       | 16            | 16               | 16                   | 0                      | 1.0000                   | 1.0000                    |
| [16](https://drive.google.com/file/d/1-0x2RXGCpw9eT5teFGixug3kUCWY6NAK/view?usp=sharing)                       | 7             | 7                | 7                    | 0                      | 1.0000                   | 1.0000                    |
| **Total**                                                                                                      | **202**       | **210**          | **202**              | **8**                  | **1.0000**               | **0.9619**                |

</div>

### Successful and unsuccessful throws classification

The table below provides a detailed assessment of the algorithm’s ability to classify both successful and
unsuccessful basketball shots using four key metrics: accuracy, precision, recall, and F1 score. 

<div align="center">

| Video Number                                                                                                   | Accuracy | Recall  | Precision | F1     |
|----------------------------------------------------------------------------------------------------------------|----------|---------|-----------|--------|
| [1](https://drive.google.com/file/d/1kv1U0XfDP2L4RrdXN8_xql9yDS44q9dT/view?usp=sharing)                         | 1.0000   | 1.0000  | 1.0000    | 1.0000 |
| [2](https://drive.google.com/file/d/17snBxQnujO61sUlmxKxlNgP1rBCj2h16/view?usp=sharing)                         | 0.8333   | 1.0000  | 0.6667    | 0.8000 |
| [3](https://drive.google.com/file/d/1--sU_ijpy_ruj6c9xkjmVtpZc3Nf_xCr/view?usp=sharing)                         | 0.5000   | 0.4444  | 0.8000    | 0.5714 |
| [4](https://drive.google.com/file/d/1-1qOl0TAYmaw8fZSJS7qzXbEC7vYjxS-/view?usp=sharing)                         | 0.8461   | 0.0000  | 0.0000    | 0.0000 |
| [5](https://drive.google.com/file/d/1-5lopGm0VivIYd3ZxRB9b3iTuSeLmMOe/view?usp=sharing)                         | 0.6923   | 0.5000  | 1.0000    | 0.6667 |
| [6](https://drive.google.com/file/d/1-7Bozphm-I9wkeZwVL2TKSZKOdKn5MI9/view?usp=sharing)                         | 0.7273   | 0.5000  | 0.6667    | 0.5714 |
| [7](https://drive.google.com/file/d/1-6S_47b1Vgyi4RXn6Oo3qOx7aKrziSPC/view?usp=sharing)                         | 0.8750   | 0.6667  | 1.0000    | 0.8000 |
| [8](https://drive.google.com/file/d/1-G_Y0n57JzOgMjnLtrwAT20D4-blU4yq/view?usp=sharing)                         | 0.8750   | 1.0000  | 0.6667    | 0.8000 |
| [9](https://drive.google.com/file/d/1-VjUzP2sdrspId_kutI_PjUdh781x8fA/view?usp=sharing)                         | 0.8667   | 0.7143  | 1.0000    | 0.8333 |
| [10](https://drive.google.com/file/d/1-aj94bD3daB13TeeqBy6nuGf7s-ORGJf/view?usp=sharing)                        | 0.9000   | 1.0000  | 0.8333    | 0.9091 |
| [11](https://drive.google.com/file/d/1-P1X2uNg3IS9zC0p0alpWN_St3a05O67/view?usp=sharing)                        | 0.9286   | 1.0000  | 0.8000    | 0.8889 |
| [12](https://drive.google.com/file/d/1-UmCmBwb-0FpceBTcsHYTaCV9dd7XRMT/view?usp=sharing)                        | 0.8571   | 1.0000  | 0.5000    | 0.6667 |
| [13](https://drive.google.com/file/d/1HBb4dpWTUzFEfrwDjm2L80TnYY4_F5t2/view?usp=sharing)                        | 0.9474   | 1.0000  | 0.9091    | 0.9524 |
| [14](https://drive.google.com/file/d/1-4hqrJqYpzQL95w_gxMCVZy2Dc6NZI9e/view?usp=sharing)                        | 0.9231   | 1.0000  | 0.8333    | 0.9091 |
| [15](https://drive.google.com/file/d/1-u_fX8oM6WtzA1Hdb3LJiKRwcIyLXvur/view?usp=sharing)                        | 0.9375   | 1.0000  | 0.8889    | 0.9412 |
| [16](https://drive.google.com/file/d/1-0x2RXGCpw9eT5teFGixug3kUCWY6NAK/view?usp=sharing)                        | 1.0000   | 1.0000  | 1.0000    | 1.0000 |
| **Total**                                                                                                       | **0.8614**| **0.8140**| **0.8537**| **0.8333**|

</div> 

## Algorithm Evaluation Conclusion
- **High throw detection accuracy**: Perfect identification accuracy (100%) across all videos, with a precision of 96.19%.
- **Challenges in outdoor environments**: Slight over-prediction due to distractions, particularly in outdoor settings.
- **Score classification**: Averaged 80.16% accuracy, with some videos performing flawlessly, while others, like Video 4, had significant misclassifications.
- **Misses classification**: Overall accuracy of 88.72%, but some inconsistencies in challenging videos.
- **Performance variability**: Indoor videos, tripod-mounted cameras, and right-side camera positions yielded better results. Free throws were easier to classify than 3-point shots.
- **Room for improvement**: Refining ball detection, shot classification, and addressing false positives—especially in outdoor environments—could enhance performance. 

## Video Requirements
1.  Each shot must end with a person touching the ball on camera, so this is considered
as the end of the throw.
2. During the throw, a person should not stand in front of the camera and obscure the
visibility of the basketball hoop.
3. If there are several basketball hoops in the video, shots and goals will be counted
in the largest hoop in the frame
4. There must be a net on the basketball hoops throughout the video.
5. There must be only 1 ball throughout the video.
6. The color of the ball must be orange or at least close to it.
7. The environment in the video can be both indoors and outdoors.
8. The camera must be stationary throughout the entire video recording.
9. The camera can be positioned on the ground or mounted at some height using a
tripod.
10. The camera must be in positions on the court marked with red dots (Picture below)

<p align="center">
  <img src="pics_readme/court_schema.png" width="600" title="Required camera positions on the court marked with red dots">
</p>
<p align="center"><em>Required camera positions on the court marked with red dots</em></p>


# Web App
## Main Technologies
Following technologies have been used in creating this web application:
- FastAPI
- FastAPI-Users
- JWT (JSON Web Tokens)
- Uvicorn
- Asyncio
- SQLAlchemy
- Pydantic
- PostgreSQL
- HTML
- CSS
- JavaScript

## Functionality + Screenshots
###  Authentication

![Login Page](pics_readme/login.png)
<p align="center"><em>Login: Secure access to the user's account using email and password.</em></p>

![Registration Page](pics_readme/register.png)
<p align="center"><em>Registration: Users can register on the web-platform using their username, email and password.</em></p>

###  Header
There are 4 buttons on the left side of the header that help users to navigate through the web app
and describe in detail about its capabilities:

####  Home Page
Main page of the application where all the features are presented

![Homepage](pics_readme/homepage.png)
<p align="center"><em>Homepage: Homepage when user is not logged in</em></p>

![Homepage](pics_readme/homepage_loggedin.png)
<p align="center"><em>Homepage: Homepage when users are logged in (their username is shown in the header)</em></p>

####  "About" Page 
Takes you to the page that describes in detail the functionality of the web app, what are the different buttons responsible for and video requirements for users.
![About](pics_readme/about.png)
<p align="center"><em>About: "About" page</em></p>

####  "Developers" Page 
Takes you to the page that is intended mainly for developers where the details
and features of the development of this application will be described (e.g. API documentation,
database setup, etc.).
![Developers](pics_readme/developers.png)
<p align="center"><em>Developers: "Developers" page</em></p>

####  "Blog" Page 
Takes you to the page that contains the links to GitHub repository of this project and
student’s LinkedIn.
![Blog](pics_readme/blog.png)
<p align="center"><em>Blog: "Blog" page</em></p>

###  Functionality
There are 3 tabs on the homepage users can access after logging in to their account:

####  Free Throws
On this page users can upload their videos of taking free throw shots to the server,
where they are processed by custom YOLOv8 deep learning model in combination with unique
algorithm for identifying the successful shots. While the video is being uploaded and processed,
users can track its progress (in %) on an informative progress bar. Users can also open the table
with statistics of all uploaded free throws videos and for each of them users can observe following
information:

| Video Name | Number of Throws | Number of Successful Shots | Number of Misses | Shooting Accuracy | Upload Date |
|------------|------------------|----------------------------|------------------|-------------------|-------------|
|            |                  |                            |                  |                   |             |

The ability is enabled for users to open the uploaded video after processing by algorithms and
see exactly how the calculating the number of goals and shots is going. Also, users can delete
uploaded videos from the list, and they will also be deleted on the server.

![Free Throws](pics_readme/freethrows.png)
<p align="center"><em>Free Throws: "Free Throws" page</em></p>

![Free Throws](pics_readme/upload_video.png)
<p align="center"><em>Free Throws: Video uploading stage</em></p>

![Free Throws](pics_readme/message.png)
<p align="center"><em>Free Throws: Message with performance stats after the video was uploaded</em></p>

####  3-Point Throws
This page is almost the same as for free throws, but it is designed to upload
videos with three-point shots. It was decided to separate these 2 types of shots since the difficulty
of successful shot of these 2 types of throws is very different: scoring a 3-point shot is much more
difficult. So, it’s more convenient to track shooting accuracy of them separately.

![3-Point Throws](pics_readme/3pthrows.png)
<p align="center"><em>3-Point Throws: "3-Point Throws" page</em></p>

####  Analyze
This page is an information panel that was created to help users better track their
basketball shooting performance over a specified period through interactive visualization. The
user can filter and view data based on different types of shots (Free Throws, 3-Point Throws, or
All) and can choose the method of filtering data either by date range or number of recent videos.
The visualization consists of the following:
1. Bar chart (Total Goals and Misses): this bar chart displays the distribution of successful
goals versus missed shots over the selected time period.
2. Line Chart (Shooting accuracy over time): this line chart tracks the accuracy of shots over
the selected date range.

![Dashboard](pics_readme/dashboard_empty.png)
<p align="center"><em>Dashboard: Dashboard while no videos uploaded</em></p>

![Dashboard](pics_readme/dashboard_1.png)
<p align="center"><em>Dashboard: Dashboard for free throws with "Date Range" filtering method</em></p>

![Dashboard](pics_readme/dashboard_2.png)
<p align="center"><em>Dashboard: Dashboard for 3-point throws with "Recent Videos" filtering method</em></p>

## Database
**PostgreSQL** database management system is used in combination with:
- **Alembic:** A database migration tool for Python.
- **SQLAlchemy:** A Python SQL toolkit and Object-Relational Mapping (ORM) library.

### Tables

#### alembic version
This table is used to store the current version of the database schema. This
version corresponds to the most recent Alembic migration that has been applied to the database.

| Column Name | Primary Key | Data Type      | Description                                                                         |
|-------------|-------------|----------------|-------------------------------------------------------------------------------------|
| version_num | Yes          | varchar(32)    | Column stores a unique identifier (usually a string of characters) representing the specific migration version. |

#### role  
This table defines the different types of roles available in the role-based access control
system of users for the web application, such as ’user’ and ’admin’. Each role corresponds to a
specific set of permissions or access levels.

| Column Name | Primary Key | Data Type   | Description                                                                                                   |
|-------------|-------------|------------|---------------------------------------------------------------------------------------------------------------|
| id          | Yes         | integer    | Column uniquely identifies each role, serving as the primary key.                                              |
| name        | No          | varchar    | Stores the name of the role, such as 'user' or 'admin', used to label the role and define permissions or access levels. |
| permission  | No          | json       | Stores a JSON object that defines specific permissions associated with the role.                               |

#### user
This table manages user information for the web application.

| Column Name      | Primary Key | Data Type                    | Description                                                                                           |
|------------------|-------------|------------------------------|-------------------------------------------------------------------------------------------------------|
| id               | Yes         | integer                      | Unique identifier for each user in the table, serving as the primary key.                              |
| email            | No          | varchar                      | Stores the email address of the user.                                                                 |
| username         | No          | varchar                      | Stores the username chosen by the user.                                                               |
| registered_at    | No          | timestamp without time zone   | Records the date and time when the user registered on the platform (UTC time zone).                    |
| role_id          | No          | integer (Foreign Key)         | References the `id` column in the "role" table, linking each user to a specific role (e.g., user, admin). |
| hashed_password  | No          | varchar                      | Stores the hashed version of the user’s password to secure it, not storing in plain text.              |
| is_active        | No          | boolean                      | Indicates whether the user’s account is active. An active account allows the user to log in and use the application. |
| is_superuser     | No          | boolean                      | Determines whether the user has superuser privileges, which includes full administrative access.        |
| is_verified      | No          | boolean                      | Indicates whether the user’s email has been verified.                                                  |

#### video
The table stores information about uploaded videos from all users including score performance statistics captured after processing the video by web application.

| Column Name      | Primary Key | Data Type                    | Description                                                                                           |
|------------------|-------------|------------------------------|-------------------------------------------------------------------------------------------------------|
| id               | Yes         | integer                      | Unique identifier for each video in the table, serving as the primary key.                             |
| name_video       | No          | varchar                      | The name of the video that the user has uploaded.                                                     |
| video_size       | No          | integer                      | The size of the uploaded video in bytes.                                                              |
| number_throws    | No          | integer                      | The number of basketball shots made in the video, calculated by the "Throws and Goals Detection" algorithm. |
| number_goals     | No          | integer                      | The number of successful basketball shots made in the video, calculated by the "Throws and Goals Detection" algorithm. |
| number_misses    | No          | integer                      | The number of missed basketball shots in the video, calculated by the "Throws and Goals Detection" algorithm. |
| accuracy         | No          | integer                      | The shooting accuracy in the video, calculated as: `accuracy = round((number_goals / number_throws) * 100)`. If `number_throws = 0`, accuracy is 0. |
| throw_type       | No          | integer                      | The type of basketball shots in the video. Can only be "free-throws" or "three-point-throws".           |
| date_upload      | No          | timestamp without time zone   | Records the date and time when the video was uploaded to the platform (UTC time zone).                 |
| user_id          | No          | integer (Foreign Key)         | References the `id` column in the "user" table, linking the video to a specific user.                  |
| video_path       | No          | varchar                      | The path to the directory where the uploaded video is stored within the project root folder.           |

## API documentation

### Interactive API Docs
This application provides automatically generated interactive API documentation, accessible through the following paths:

- **Swagger UI**: Available at `/docs`, this interface allows users to interact with the API endpoints directly from the browser. It provides a user-friendly way to explore the API, send requests, and view responses.
- **ReDoc**: Available at `/redoc`, this interface offers a more detailed and structured view of the API documentation.

### Endpoint Overview
The API is divided into different groups based on functionality:

**1. General Routes**
- **`/dashboard-redirect`** (GET): Redirect to Dash App with transferring Cookies. Authentication is required (JWT).
  - *Common errors:*
    - *Code 401* (Unauthorized if JWT token is missing or invalid)

- **`/free-throws`** (GET): Get Free Throws Page (free_throws.html)

- **`/three-point-throws`** (GET): Get Three Point Throws Page (three_point_throws.html)

- **`/register`** (GET): Get Registration Page (register.html)

- **`/login`** (GET): Get Login Page (login.html)

- **`/homepage`** (GET): Get Home Page (homepage.html)

- **`/about`** (GET): Get About Page (about.html)

- **`/developers`** (GET): Get Developers Page (developers.html)

- **`/blog`** (GET): Get Blog Page (blog.html)

- **`/protected-route`** (GET): Return username of the current user. Authentication is required (JWT).
  - *Common errors:*
    - *Code 401* (Unauthorized if JWT token is missing or invalid)

**2. Authentication Routes**
- **`/auth/jwt/login`** (POST): Authenticates a user, giving a JWT token.
  - **Request Body:**
    ```json
    {
        "username": "user@example.com",
        "password": "string"
    }
    ```
  - **Successful Response:**
    - *Code 204* (No Content)
  
  - *Common errors:*
    - *Code 400* (Error: Bad Request, if user with such username and password is not found in the database):
      ```json
      {
          "detail": "LOGIN_BAD_CREDENTIALS"
      }
      ```

- **`/auth/jwt/logout`** (POST): User logout using JWT. Authentication is required (JWT).
  - **Successful Response:**
    - *Code 204* (No Content)

  - *Common errors:*
    - *Code 401* (Error: Unauthorized, if user is not authorized):
      ```json
      {
          "detail": "Unauthorized"
      }
      ```

- **`/auth/forgot-password`** (POST): Request password reset (not used in my application).

- **`/auth/reset-password`** (POST): Reset password using a token (not used in my application).

- **`/auth/register`** (POST): Registers a new user.
  - **Request Body:**
    ```json
    {
        "username": "string",
        "email": "user@example.com",
        "password": "string"
    }
    ```
  - **Successful Response:**
    ```json
    {
        "id": 1,
        "email": "user@example.com",
        "is_active": true,
        "is_superuser": false,
        "is_verified": false,
        "username": "string",
        "role_id": 1
    }
    ```
  - *Common errors:*
    - *Code 400* (Error: Bad Request, if user already exists):
      ```json
      {
          "detail": "REGISTER_USER_ALREADY_EXISTS"
      }
      ```

**3. Video Management Routes**
- **`/videos/statistics/{throw_type}`** (GET): Retrieves video statistics for the given `throw_type`.
  - **Path Parameters:**
    - `throw_type` (string):
      - `free-throw`: for videos containing free throws.
      - `three-point-throws`: for videos containing 3-point throws.
      - `all`: for videos containing both types of throws.
  
  - **Successful Response** (Example: `throw_type = free-throws`):
    ```json
    [
        {
            "id": 11,
            "name_video": "yarik_tripid_throws_1.mp4",
            "video_size": 1125993,
            "number_throws": 2,
            "number_goals": 0,
            "number_misses": 2,
            "accuracy": 0,
            "throw_type": "free-throws",
            "date_upload": "2024-07-29T23:17:29.468100",
            "user_id": 1,
            "video_path": "result_videos\\result_yarik_tripid_throws_1.mp4"
        },
        {
            "id": 12,
            "name_video": "yarik_tripid_throws_2.mp4",
            "video_size": 1663929,
            "number_throws": 1,
            "number_goals": 1,
            "number_misses": 0,
            "accuracy": 100,
            "throw_type": "free-throws",
            "date_upload": "2024-07-29T23:38:40.071928",
            "user_id": 1,
            "video_path": "result_videos\\result_yarik_tripid_throws_2.mp4"
        }
    ]
    ```
  - *Common Errors:*
    - *Code 401* (Unauthorized if the user is not authorized):
      ```json
      {
          "detail": "Unauthorized"
      }
      ```

- **`/videos/upload_video_endpoint/{throw_type}`** (POST): Upload video on the platform and start video processing. Authentication is required (JWT).
  - **Path Parameters:**
    - `throw_type` (string):
      - `free-throw`: for videos containing free throws.
      - `three-point-throws`: for videos containing 3-point throws.
      - `all`: for videos containing both types of throws.

  - **Successful Response**:
    ```json
    {
        "success": true,
        "message": "Video processing started",
        "task_id": "256b09dc-93e5-4ecf-9b56-42fa95503d20"
    }
    ```

  - *Common Errors:*
    - *Code 400* (Error: Bad Request, if user chooses file not in `.mp4` format):
      ```json
      {
          "detail": "Invalid file format. Acceptable format .mp4"
      }
      ```
    - *Code 401* (Unauthorized if the user is not authorized):
      ```json
      {
          "detail": "Unauthorized"
      }
      ```

- **`/videos/get_video_processing_results/{task_id}`** (GET): Get results of a video processing task.
  - **Path Parameters:**
    - `task_id` (string): ID of the video processing task.
  
  - **Successful Response** (Example: `task_id = 256b09dc-93e5-4ecf-9b56-42fa95503d20`):
    ```json
    {
        "success": true,
        "message": "Video processing completed",
        "score": 1,
        "misses": 0,
        "throws": 1
    }
    ```

  - *Common Errors:*
    - *Code 404* (Task ID not found):
      ```json
      {
          "detail": "Task ID not found"
      }
      ```

- **`/videos/delete_video/{video_path}`** (DELETE): Delete a video from the database and the directory where processed videos are stored.
  - **Path Parameters:**
    - `video_path` (string): The path to the video file.
  
  - **Successful Response**:
    ```json
    {
        "success": true
    }
    ```

  - *Common Errors:*
    - *Code 404* (Video not found):
      ```json
      {
          "detail": "Video not found"
      }
      ```

- **`/videos/stop_video_processing/{task_id}`** (POST): Stop video processing task. Used when resetting the task (e.g., wrong file upload).
  - **Path Parameters:**
    - `task_id` (string): ID of the video processing task.
  
  - **Successful Response** (Example: `task_id = e02bad05-ed31-472d-a0f8-aad83d3a336a`):
    ```json
    {
        "success": true,
        "message": "Video processing stopped"
    }
    ```

  - *Common Errors:*
    - *Code 404* (Task ID not found):
      ```json
      {
          "detail": "Invalid task_id"
      }
      ```

## Authentication In Web Application
### Overview
In this web application, user authentication and management are handled using the fastapi-users
python library. This library simplifies the integration of user authentication, including handling user
registration, login, and password management, with built-in support for JWT (JSON Web Token)
authentication.

### JWT Authentication Strategy
The application utilizes JWT (JSON Web Tokens) for authenticating users. JWT is a compact and
self-contained way for securely transmitting information between parties as a JSON object.

When a user logs in through the /auth/jwt/login endpoint, their credentials (email/username and
password) are verified. Upon successful authentication, the server generates a JWT token, which
contains encoded information about the user, such as their ID, roles, and token expiration time. This
JWT token stored in the cookie is automatically transmitted with every subsequent HTTP request
from the client to the server such as API calls to fetch data, submit forms, or navigate to protected
routes. This allows the server to verify the user’s identity and permissions for each request. The token
is included in the header of HTTP requests, formatted as follows (example):

```json
cookie : basketball = eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxIiwiYXVkIjpbImZhc3RhcGktdXNlcnM6YXV0aCJdLCJleHAiOjE3MjQyNTM0MDl9.7W_5BlnIb1FVRHk6bqz8flCNKRlH8DaIEN8nHN3Vum0
```

When the server receives a request, it checks the JWT token in the cookie to ensure that it is
valid (e.g. not expired, correctly signed, etc.). If the token is valid, the server identifies the user and
processes the request according to the user’s permissions and roles. If the token is invalid or expired,
the server will return an error response, typically a 401 Unauthorized status, prompting the user to
log in again.

# Conclusion
- **Primary research question answered**: Successfully integrated a custom-trained YOLOv8 model into a web application to detect and analyze basketball shooting activities.
- **Model performance**: High precision and recall in detecting players, balls, and hoops, though challenges like occlusion remain.
- **Throw and goal detection**: Reliable algorithm performance, with room for improvement in handling occlusions and challenging conditions.
- **Web application development**: FastAPI-based platform efficiently processes videos and provides shot statistics, meeting the project’s objective.
- **Future research**: The impact of quantitative feedback on training habits was not fully explored and warrants further study.
- **Overall contribution**: Significant achievement in applying computer vision to basketball training, providing a solid foundation for future advancements in sports analytics.