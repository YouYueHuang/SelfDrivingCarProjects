## Finding Lane Lines on the Road

<p align="center">
  <b>Lane with solid left yellow line</b><br>
	<a href="https://www.youtube.com/watch?v=8WpxG8XdfZY"><img src="./img/solidYellowLeft.gif" alt="Overview" width="48%" height="275px"></a>
	<a href="https://www.youtube.com/watch?v=1rsAzuuPxj0"><img src="./img/colorFilter_solidYellowLeft.gif" alt="Overview" width="48%" height="275px"></a>
  <br>
  <b>Lane with right solid white line</b><br>
	<a href="https://www.youtube.com/watch?v=8WpxG8XdfZY"><img src="./img/solidWhiteRight.gif" alt="Overview" width="48%" height="275px"></a>
	<a href="https://www.youtube.com/watch?v=A6O8dB1wdZ4"><img src="./img/colorFilter_solidWhiteRight.gif" alt="Overview" width="48%" height="275px"></a>
  <br>
  <b>Lane with shadow</b><br>
	<a href="https://www.youtube.com/watch?v=8WpxG8XdfZY"><img src="./img/lane_line_logo.gif" alt="Overview" width="48%" height="275px"></a>
	<a href="https://www.youtube.com/watch?v=BYAPi9Xv6cs"><img src="./img/colorFilter_challenge.gif" alt="Overview" width="48%" height="275px"></a>
  <br>
</p>

### Overview
---
To develop a self-driving car, one of the critical issues is to tell the car where to go. The lines on the road that show drivers where the lanes act as a constant reference for where to steer the vehicle. This project builds an algorithm and applies Python packages to automatically detect lane lines in images.

### Getting Started
---
* [Main_pipeline.ipynb](https://github.com/YouYueHuang/SelfDrivingCarProjects/tree/master/CarND-LaneLines-P1-master/Main_pipeline.ipynb) 
This notebook contains the code to detect lane line and visualizes the processing steps.

* [examples](https://github.com/YouYueHuang/SelfDrivingCarProjects/tree/master/CarND-LaneLines-P1-master/examples) 
This directory stores the results of each step in image processing pipeline.

* [test_images](https://github.com/YouYueHuang/SelfDrivingCarProjects/tree/master/CarND-LaneLines-P1-master/test_images), [test_images_output](https://github.com/YouYueHuang/SelfDrivingCarProjects/tree/master/CarND-LaneLines-P1-master/test_images_output)
This directory contains the test images and the detected lane lines.

* [test_videos](https://github.com/YouYueHuang/SelfDrivingCarProjects/tree/master/CarND-LaneLines-P1-master/test_videos), [test_videos_output](https://github.com/YouYueHuang/SelfDrivingCarProjects/tree/master/CarND-LaneLines-P1-master/test_videos_output) 
This directory contains the test videos and the detected lane lines.

### Pipeline
---
The following steps are listed based on the image processing order. The images are the results of that step. 

<p align="center">
  <b>Original image</b><br>
  <img src="./img/input.jpg" alt="Overview" width="60%" height="300px">
</p>

<p align="center">
  <b>Select region of interest</b><br>
  <img src="./img/region_of_interest.jpg" alt="Overview" width="60%" height="300px">
</p>

* The code for selecting region of interest

```python
vertices = np.array([[(0, img_size[0])
                     , (img_size[1]//2, img_size[0]//2)
                     , (img_size[1], img_size[0])]], dtype=np.int32)

mask = np.zeros_like(img)   
if len(img.shape) > 2:
    channel_count = img.shape[2] 
    ignore_mask_color = (255,) * channel_count
else:
    ignore_mask_color = 255

cv2.fillPoly(mask, vertices, ignore_mask_color)
masked_image = cv2.bitwise_and(img, mask)
```

<p align="center">
  <b>Scatter plot of pixel distribution in RGB color space</b>
</p>
<table>
  <tr>
    <td align="center"><b>GB distribution</b></td>
    <td align="center"><b>RB distribution</b></td>
    <td align="center"><b>RG distribution</b></td>
  </tr> 
  <tr>
    <td><img src="./img/GB_distribution.png" alt="Overview"></td>
    <td><img src="./img/RB_distribution.png" alt="Overview"></td>
    <td><img src="./img/RG_distribution.png" alt="Overview"></td>
  </tr>
</table>

<p align="center">
  <b>Color Filtering</b><br>
  <a href="https://youtu.be/8WpxG8XdfZY">video</a><br>
  <img src="./img/lane_line.jpg" alt="Overview" width="60%" height="300px">
</p>

* The code for selecting yellow and white lane line

```python
yellowLine_color_boundary = ([200,170,0], [255, 225, 150])
WhiteLine_color_boundary = ([210,210,210], [255, 255, 255])

# it can be replaced with WhiteLine_color_boundary
lower, upper = yellowLine_color_boundary 

# create NumPy arrays from the boundaries
lower = np.array(lower, dtype = "uint8")
upper = np.array(upper, dtype = "uint8")

# find the colors within the specified boundaries and apply the mask
mask = cv2.inRange(img, lower, upper)
filtered_img = cv2.bitwise_and(img, img,  mask= mask)
```

<p align="center">
  <b>Gray scale transformation</b><br>
  <img src="./img/contrast_enhanced_line.jpg" alt="Overview" width="60%" height="300px">
</p>


<p align="center">
  <b>Canny edge detection</b><br>
  <img src="./img/canny_img.jpg" alt="Overview" width="60%" height="300px">
</p>

* The code for Canny edge detection

```python
low_threshold = 50
high_threshold = 200
cv2.Canny(img, low_threshold, high_threshold)
```

<p align="center">
  <b>Gaussian blurred processing</b><br>
  <img src="./img/gussian_blurred_img.jpg" alt="Overview" width="60%" height="300px">
</p>

* The code for Gaussian blurring

```python
kernel_size = 7
cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
```

#### Hough Line Transformation
---

Increasing the threshold value will increase the minimum number of intersection required to detect a line and thus is able to differentiate between the left and right lanes better

`min_line_len` as the name suggests will help make sure that the line segments are drawn on the actual lines and thus help eliminate some of the lines from background noise.

Increasing the value of `max_line_gap` will help get more connected annotated lines when there are broken lanes as it allows points that are farther away from each other to be connected with a single line.

To detect curves and faded lanes, Hough lines algorithm could be extended to detect tangents of the curve which helps separate the left and right lines.

<p align="center">
  <b>Hough line transformation</b><br>
  <img src="./img/houghLines_img.jpg" alt="Overview" width="60%" height="300px">
</p>

* The code for Hough line transformation

```python
rho = 1
theta = np.pi/180
threshold = 50
min_line_len = 100
max_line_gap = 160

cv2.HoughLinesP(img
              , rho
              , theta
              , threshold, np.array([])
              , minLineLength=min_line_len
              , maxLineGap=max_line_gap), axis=1)
```

<p align="center">
  <b>Scatter matrix plot of intercept, length and slope of Hough lines in a video frame</b><br>
  <img src="./img/LineFeatureScattermatrix.png" alt="Overview" width="80%" height="80%">
</p>

<p align="center">
  <b>Detected lane line after extrapolation</b><br>
  <img src="./img/out_image.jpg" alt="Overview" width="60%" height="300px">
</p>

### Differences from the baselines
---
* Add slope and distance feature for removing noise. The distance is defined as the average distance of end points of a Hough line. The slope feature is the slope of a Hough line. Intercept was also considers as a filter feature, but it turn out to be potential problem. Some correct lines might be removed if the threshold is not well set. Also, the threshold is hard to be generalized. The shift of the car position on the road might lead to bad filtering resulet. For instance, Sometimes no lane line is detected.

* The algorithm assumes that longer Hough lines are more likely to be lane line segments. Based on the assumption, squared Hough line length is applied as weight to calculate the mass center of the Hough lines. The slope is the average slope of the Hough lines. The mass center and the average slope are used for lane line extrapolation.

### Limitations
---
* All parameters of the computer vision algorithms are chosen manually, so it is hard to generalize them. In the video of challenge, the lane line detection algorithm was seriously influenced by the enviornments such as the shadow of the trees, the the sun light, the stain on the road, etc. The results might be wrong if the images are from in other unknown environment such as rainy day.

* The region of interest is triangular, and most of the noise are from the top of the triangle (Not far from the road ahead.) These noise are caused by other cars or cluttered view. Hough line transformation wrongly detected these noise as possible lane line.   

### Possible improvements
---
* Sometimes Hough line transformation wrongly detected these noise as possible lane line. One improvement could be the application of linear regression. 
In this algorithm, Canny edge detection and Guassian blurring are the preparation before Hough line transformation. After series of filtering, local contrast enhancement and sharpening could be applied to replace these processings. Next, The points can be fitted into a line with linear regression to get the lane lines.

* As for the noise from objects on the road ahead, the region of interest could be changed from triangle to trapezoid. The noise could be reduced in our examples, but a good region of interest might still vary due to the angle of camara, the condition of road, etc. 

### Reference
---
[Robust and real time detection of curvy lanes (curves) having desired slopes for driving assistance and autonomous vehicles](http://airccj.org/CSCP/vol5/csit53211.pdf)