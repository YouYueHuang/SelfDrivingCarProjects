# **Finding Lane Lines on the Road** 

<img src="examples/laneLines_thirdPass.jpg" width="480" alt="Combined Image" />

Overview
---
To develop a self-driving car, one of the critical issues is to tell the car where to go. The lines on the road that show drivers where the lanes act as a constant reference for where to steer the vehicle. This project builds an algorithm and applies Python packages to automatically detect lane lines in images.

Getting Started
---
[Main_pipeline.ipynb](https://github.com/YouYueHuang/SelfDrivingCarProjects/tree/master/CarND-LaneLines-P1-master/Main_pipeline.ipynb) This notebook contains the code to detect lane line and visualizes the processing steps. It shows an example of using a model pre-trained on MS COCO to segment objects in your own images.

[examples](https://github.com/YouYueHuang/SelfDrivingCarProjects/tree/master/CarND-LaneLines-P1-master/examples) It shows the results of each step in image processing pipeline.

[test_images](https://github.com/YouYueHuang/SelfDrivingCarProjects/tree/master/CarND-LaneLines-P1-master/test_images), [test_images_output](https://github.com/YouYueHuang/SelfDrivingCarProjects/tree/master/CarND-LaneLines-P1-master/test_images_output): These files contain the test images and the detected lane lines.

[test_videos](https://github.com/YouYueHuang/SelfDrivingCarProjects/tree/master/CarND-LaneLines-P1-master/test_videos), [test_videos_output](https://github.com/YouYueHuang/SelfDrivingCarProjects/tree/master/CarND-LaneLines-P1-master/test_videos_output): These files contain the test videos and the detected lane lines.

Pipeline
---



* select color of interest

We want to be able to detect each of the Game Boy cartridges in the image. That means we’ll have to recognize red, blue, yellow, and gray colors in the image.

Let’s go ahead and define this list of colors:

<a href="https://www.youtube.com/watch?v=8WpxG8XdfZY"><img src="./img/lane_line_logo.gif" alt="Overview" width="60%" height="60%"></a>
<a href="https://www.youtube.com/watch?v=BYAPi9Xv6cs"><img src="./project_4_advanced_lane_finding/img/overview.gif" alt="Overview" width="60%" height="60%"></a>



Conclusion
---

2. Identify any shortcomings
This pipeline did not apply any machine learning techniques. All parameters of the computer vision algorithms are chosen manually. In the video of challenge, the lane line detection algorithm is influenced seriously by the enviornments such as the shadow of the trees, the the sun light, the stain on the road, etc.

Sometimes, the after Hough line detection due to the parameter setting. Once the parameter threshould 

3. Suggest possible improvements
In this project, the gray scale image step might be skipped. 255
the region of interest is triangular, but it would be good to try the Trapezoid
linear regression 
contrast enhancement