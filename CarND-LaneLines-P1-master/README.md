# **Finding Lane Lines on the Road** 
--------
Overview
---
To develop a self-driving car, one of the critical issues is to tell the car where to go. The lines on the road that show drivers where the lanes act as a constant reference for where to steer the vehicle. This project builds an algorithm and applies Python packages to automatically detect lane lines in images.

Getting Started
---
```shell
[Main_pipeline.ipynb](https://github.com/YouYueHuang/SelfDrivingCarProjects/tree/master/CarND-LaneLines-P1-master/Main_pipeline.ipynb) 
```
This notebook contains the code to detect lane line and visualizes the processing steps. It shows an example of using a model pre-trained on MS COCO to segment objects in your own images.

```shell
[examples](https://github.com/YouYueHuang/SelfDrivingCarProjects/tree/master/CarND-LaneLines-P1-master/examples) 
```
It shows the results of each step in image processing pipeline.

```shell
[test_images](https://github.com/YouYueHuang/SelfDrivingCarProjects/tree/master/CarND-LaneLines-P1-master/test_images), [test_images_output](https://github.com/YouYueHuang/SelfDrivingCarProjects/tree/master/CarND-LaneLines-P1-master/test_images_output) 
```
These files contain the test images and the detected lane lines.

```shell
[test_videos](https://github.com/YouYueHuang/SelfDrivingCarProjects/tree/master/CarND-LaneLines-P1-master/test_videos), [test_videos_output](https://github.com/YouYueHuang/SelfDrivingCarProjects/tree/master/CarND-LaneLines-P1-master/test_videos_output) 
```
These files contain the test videos and the detected lane lines.

Pipeline
---

* Original image

<img src="./examples/input.jpg" alt="Overview" width="60%" height="300px">

* Select region of interest

<img src="./examples/region_of_interest.jpg" alt="Overview" width="60%" height="300px">

* Select color of interest

<img src="./examples/GB_distribution.png" alt="Overview" width="33%" height="300px">
<img src="./examples/RB_distribution.png" alt="Overview" width="33%" height="300px">
<img src="./examples/RG_distribution.png" alt="Overview" width="33%" height="300px">

<img src="./examples/lane_line.jpg" alt="Overview" width="60%" height="300px">

* Gray scale transformation

<img src="./examples/contrast_enhanced_line.jpg" alt="Overview" width="60%" height="300px">

* Canny transformation

<img src="./examples/canny_img.jpg" alt="Overview" width="60%" height="300px">

*　Gaussian blurred processing

<img src="./examples/gussian_blurred_img.jpg" alt="Overview" width="60%" height="300px">

*　Hough line transformation

<img src="./examples/houghLines_img.jpg" alt="Overview" width="60%" height="300px">

*　Detected lane line after extrapolation

<img src="./examples/out_image.jpg" alt="Overview" width="60%" height="300px">

* Video with detected lane line

```shell
solidYellowLeft.mp4
```
<a href="https://www.youtube.com/watch?v=8WpxG8XdfZY"><img src="./img/solidYellowLeft.gif" alt="Overview" width="45%" height="300px"></a>
<a href="https://www.youtube.com/watch?v=1rsAzuuPxj0"><img src="./img/colorFilter_solidYellowLeft.gif" alt="Overview" width="50%" height="300px"></a>

```shell
solidWhiteRight.mp4
```
<a href="https://www.youtube.com/watch?v=8WpxG8XdfZY"><img src="./img/solidWhiteRight.gif" alt="Overview" width="45%" height="300px"></a>
<a href="https://www.youtube.com/watch?v=A6O8dB1wdZ4"><img src="./img/colorFilter_solidWhiteRight.gif" alt="Overview" width="50%" height="300px"></a>

```shell
challenge.mp4
```
<a href="https://www.youtube.com/watch?v=8WpxG8XdfZY"><img src="./img/lane_line_logo.gif" alt="Overview" width="45%" height="300px"></a>
<a href="https://www.youtube.com/watch?v=BYAPi9Xv6cs"><img src="./img/colorFilter_challenge.gif" alt="Overview" width="45%" height="300px"></a>

Differences from the baselines
---
1. Add color filters for yellow and white lane lines.

2. Add slope and distance feature for removing noise. The distance is defined as the average distance of end points of a Hough line. The slope feature is the slope of a Hough line. Intercept was also considers as a filter feature, but it turn out to be potential problem. Some correct lines might be removed if the threshold is not well set. Also, the threshold is hard to be generalized. The shift of the car position on the road might lead to bad filtering resulet. For instance, Sometimes no lane line is detected.

3. The algorithm assumes that longer Hough lines are more likely to be lane line segments. Based on the assumption, squared Hough line length is applied as weight to calculate the mass center of the Hough lines. The slope is the average slope of the Hough lines. The mass center and the average slope are used for lane line extrapolation.

Limitations
---
All parameters of the computer vision algorithms are chosen manually, so it is hard to generalize them. In the video of challenge, the lane line detection algorithm was seriously influenced by the enviornments such as the shadow of the trees, the the sun light, the stain on the road, etc. The results might be wrong if the images are from in other unknown environment such as rainy day.

The region of interest is triangular, and most of the noise are from the top of the triangle (Not far from the front of the car.) These noise are caused by other cars or cluttered view. Hough line transformation wrongly detected these noise as possible lane line.   

Possible improvements
---
Sometimes Hough line transformation wrongly detected these noise as possible lane line. One improvement could be the application of linear regression. 
In this algorithm, Canny detection and Guassian blurring are the preparation before Hough line transformation. After series of filtering, local contrast enhancement and sharpening could be applied to replace these processings. Next, The points can be fitted into a line with linear regression to get the lane lines.

As for the noise from the front of the car, the region of interest could be changed from triangle to trapezoid. The noise could be reduced in our examples, but a good region of interest might still vary due to the angle of camara, the condition of road, etc. 
