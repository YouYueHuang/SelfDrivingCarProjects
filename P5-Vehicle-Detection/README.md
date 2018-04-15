## Vehicle Detection

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

### Overview
---
In this project, the goal is to build a software pipeline to detect vehicles in a video (start with the `test_video.mp4` and later implement on full `project_video.mp4`).

### Project goal
---
The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

### Datasets
---
There are two labeled data to train the classifier
[vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip)
[non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip). 

Some example images for testing your pipeline on single frames are located in the `test_images` folder.  To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `ouput_images`, and include them in your writeup for the project by describing what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  


### Histogram of Oriented Gradients (HOG)
---

1. I used the `skimage.feature.hog()` to extracted HOG features just once for the entire region of interest in each full image / video frame. 
In the beginning, 
Explain how you settled on your final choice of HOG

```python
from skimage.feature import hog
features = hog(img, orientations=orient, 
               pixels_per_cell=(pix_per_cell, pix_per_cell),
               cells_per_block=(cell_per_block, cell_per_block), 
               transform_sqrt=True, 
               visualise=False, feature_vector=False)
```

Explanation given for methods used to extract HOG features, including which color space was chosen, which HOG parameters (orientations, pixels_per_cell, cells_per_block), and why.


Answer: 

2. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The HOG features extracted from the training data have been used to train a classifier, could be SVM, Decision Tree or other. Features should be scaled to zero mean and unit variance before training the classifier.


Answer: normalization the training data
Use sklearn.preprocessing.StandardScaler() to normalize your feature vectors for training your classifier as described in this lesson. Then apply the same scaling to each of the feature vectors you extract from windows in your test images.



For feature extraction, I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:



#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:



Result
1. I tried to compared the Hog features of a car and a not car example based on mean absolute difference. Although the result shows the car image with less features has larger mean abosulte differece over all pixels, the accuracy of classfier did not show better performance
2. I rescaled the window with manually, and it can detect the cars successfully.

Conclusion:
1. Different sets of features (HOG, spatial binning of color, histogram of color) could be tested separately based on different color spaces (RGB, HSV, LUV, HLS, YUV, YCrCb, LAB)
2. I expect different window size, large van, hides the sky. It might fail to 
3. Augment data with disappeard low half part
4. Dectec all the lane line and perspective transform and define the window because in real case, we still need to detect lane line. The lane line can be used to estimate the size and aspect ratio of the window.