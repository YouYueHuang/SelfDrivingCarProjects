## Advanced Lane Finding

[![Video](./imgs/bird_view_project_video.gif)](#)
[![Video](./imgs/perspective_view_project_video.gif)](#)

[//]: # (Image References)

[image1]: ./imgs/pipeline.png "pipeline"
[image2]: ./imgs/warped_straight_lines.jpg "warped straight lines"
[image3]: ./imgs/origin_test_images.png "origin test images"
[image4]: ./imgs/undistorted_test_images.png "undistorted test images"
[image5]: ./imgs/perspective_transform_test_images.png "perspective transform test images"
[image6]: ./imgs/Contrast_Enhancement_test_images.png "Contrast enhanced test images"
[image7]: ./imgs/Gradient_threshold_test_images.png "Gradient threshold"
[image8]: ./imgs/Gradient_Magnitude_threshold_test_images.png "Gradient Magnitude threshold"
[image9]: ./imgs/Gradient_Direction_test_images.png "Gradient Direction threshold"
[image10]: ./imgs/HLS_threshold_test_images.png "HLS threshold"
[image11]: ./imgs/HSV_threshold_test_images.png "HSV threshold"
[image12]: ./imgs/RGB_threshold_test_images.png "RGB threshold"
[image13]: ./imgs/filters_overview_test_images.png "filters overview"
[image14]: ./imgs/histogram_peak_detection_test_images.png "histogram peak detection test images"
[image15]: ./imgs/fit_lane_with_sliding_window.png "fit lane with sliding window"
[image16]: ./imgs/sliding_window_test_images.png "sliding window"
[image17]: ./imgs/color_fit_lines.jpg "color fit lines"
[image18]: ./imgs/draw_lanes_bird_view_test_images.png "draw lanes from bird view"
[image19]: ./imgs/draw_lanes_perspective_view_test_images.png "draw lanes from perspective view"
[image20]: ./imgs/bird_view_project_video.gif "bird view project video"
[image21]: ./imgs/perspective_view_project_video.gif "perspective view project video"

### Overview
---------------
When we drive on the road, the lines on the road act as the references. To develope a self-driving car, one of the significant goals is to detect lane and automatically steer the vehicle.

In this project, the main goal is to build a pipeline to identify the lane boundaries in a video using cmputer vision techniques starting from camera caliaration to detect lane lines and calculate some characteristic of these lanes.

### Goal
---------------
The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to the images (or videos).
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

### Files
---------------

The structure and useage of the files in this repository are as follows:

`main_pipeline.ipynb`: This part contains all the utility, visualization, preprocessing, computer vision techniques and vedio processing codes.

`test_images`: It contains 11 images extracted from the test videos.

`test_videos`: It contains 3 test videos. The output.

`camera_cal`: It contains chessboard images for clibration.

`img`: It stores the results during image processing.

`utils`: It contains the functions used for image processing in this project.

#### Pipeline (test images)
------------------
1. Provide an example of a distortion-corrected image.

Distortion correction that was calculated via camera calibration has been correctly applied to each image. An example of a distortion corrected image should be included in the writeup (or saved to a folder) and submitted with the project.


2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.

A method or combination of methods (i.e., color transforms, gradients) has been used to create a binary image containing likely lane pixels. There is no "ground truth" here, just visual verification that the pixels identified as part of the lane lines are, in fact, part of the lines. Example binary images should be included in the writeup (or saved to a folder) and submitted with the project.

3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

OpenCV function or other method has been used to correctly rectify each image to a "birds-eye view". Transformed images should be included in the writeup (or saved to a folder) and submitted with the project.

4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Methods have been used to identify lane line pixels in the rectified binary image. The left and right line have been identified and fit with a curved functional form (e.g., spine or polynomial). Example images with line pixels identified and a fit overplotted should be included in the writeup (or saved to a folder) and submitted with the project.

5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Here the idea is to take the measurements of where the lane lines are and estimate how much the road is curving and where the vehicle is located with respect to the center of the lane. The radius of curvature may be given in meters assuming the curve of the road follows a circle. For the position of the vehicle, you may assume the camera is mounted at the center of the car and the deviation of the midpoint of the lane from the center of the image is the offset you're looking for. As with the polynomial fitting, convert from pixels to meters.

6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The fit from the rectified image has been warped back onto the original image and plotted to identify the lane boundaries. This should demonstrate that the lane boundaries were correctly identified. An example image with lanes, curvature, and position from center should be included in the writeup (or saved to a folder) and submitted with the project.



The images for camera calibration are stored in the folder called `camera_cal/calibration_mat.p`. The images in `test_images` are for testing pipeline on single frames.  

If you want to extract more test images from the videos, you can simply use an image writing method like `cv2.imwrite()`, i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.  

The video called `project_video.mp4` is the video your pipeline should work well on.  
The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions.  
The `harder_challenge.mp4` video is another optional challenge and is brutal!


Result:
1. multiple lines
2. the noise from  edge of fence, shade, dirt, stains, repainted asphalt
3. the line on the far side is unclear
4. weighted point
5. window size
6. weighted filter
7. enhance the contrast of image with LAB color space and 
8. the colors were changed because of shade and dirt
9. the maginitude of dash line in histogram is less than that of edge.
10. low curvature need wider windows
11. regard the lines as assisting lines, not noises 




Conclusion:
1. Tracking 
2. pipeline should be modified. detect starting position and search hot region
if the lane line disapear
3. Smoothing: Even when everything is working, your line detections will jump around from frame to frame a bit and it can be preferable to smooth over the last n frames of video to obtain a cleaner result. Each time you get a new high-confidence measurement, you can append it to the list of recent measurements and then take an average over n past measurements to obtain the lane position you want to draw onto the image.


Camera Calibration
state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

OpenCV functions or other methods were used to calculate the correct camera matrix and distortion coefficients using the calibration chessboard images provided in the repository (note these are 9x6 chessboard images, unlike the 8x6 images used in the lesson). The distortion matrix should be used to un-distort one of the calibration images provided as a demonstration that the calibration is correct. Example of undistorted calibration image is Included in the writeup (or saved to a folder).



