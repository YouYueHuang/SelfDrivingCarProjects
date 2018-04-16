## Vehicle Detection

[//]: # (Image References)
[image1]: ./img/cars.png
[image2]: ./img/not_cars.jpg
[image3]: ./img/HOG_hyperparameter_visualization.jpg
[image4]: ./img/bin_spatial.jpg
[image5]: ./img/color_of_histogram.png
[image6]: ./img/histogram_of_color_space.png.png
[image7]: ./img/img_different_color_space.png
[image8]: ./img/color_distribution_of_different_color_space.png
[image9]: ./img/sliding_window.png
[image10]: ./img/heatmap.png.png
[image11]: ./img/output_bboxes.png
[image12]: ./img/HOG_hyperparameter_test.png
[image13]: ./img/image0004.png




### Overview
---
In this project, the goal is to build a software pipeline to detect vehicles in a video (start with the `test_video.mp4` and later implement on full `project_video.mp4`).

### Project goal
---
The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images 
* Apply a color transform to extract binned color features, as well as histograms of color. 
* Train a classifier Linear SVM classifier
* Estimate a bounding box for vehicles detected.
* Implement a sliding-window technique and search for vehicles of a image with trained classifier.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to locate vehicles.

### File structure
---
The structure and useage of the files in this repository are as follows:

`Main_pipeline.ipynb`: This part contains the code for feature extraction, preprocessing, classifier training and sliding window.

`test_images`: It contains images for testing feature extraction, sliding window drawing and .

`videos`: It contains test videos `project_video.mp4` and the precessed result. 

`img`: It stores the images of README.md.

`saved_file`: It contains the `model.p` file. The model stores the parameters for feature extraction.

`color_space_exploration.py`: The function plots the distribution of the pixels over different color spaces (RGB, HLS, HSV, LAB, LUV, YUV, YCrCb)

### Datasets
---
There are two labeled data to train the classifier
  1. [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip)
  2. [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip).

Total image number: 17760
Number of car images :  8792
Number of non-car images:  8968

The following figures shows the examples of car and not-car images 
![alt text][image1]
![alt text][image2]

### Pipeline
### Feature Extraction
---

#### Histogram of Oriented Gradients (HOG)

I used the `skimage.feature.hog()` to extracted HOG features just once for the entire region of interest in each full image / video frame. 
Next, I resized the region of interest of a image to solve the car size issue. The size of training samples are 64x64, so we need to resize the samples by sliding window and fit the samples into the classifer. If the prediction is positive, then the window is stored in the candidate list.

The following figures shows the HOG features of a car and a not-car image 
![alt text][image3]

```python
from skimage.feature import hog
def get_hog_features():
	return hog(img, orientations=orient, 
               pixels_per_cell=(pix_per_cell, pix_per_cell),
               cells_per_block=(cell_per_block, cell_per_block), 
               transform_sqrt=True, 
               visualise=False, feature_vector=False)

img = img.astype(np.float32)/255

# sub-sampling
img_tosearch = img[y_start:y_stop, x_start: xstop, :]  
ctrans_tosearch = convert_color(img_tosearch, conv='RGB2{any color space}')
    
# search region of interest
if scale != 1:
    imshape = ctrans_tosearch.shape
    ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
ch1 = ctrans_tosearch[:,:,0]
ch2 = ctrans_tosearch[:,:,1]
ch3 = ctrans_tosearch[:,:,2]
    
# cells_per_step = 2 would result in a search window overlap of 75% 
# (2 is 25% of 8, so we move 25% each time, leaving 75% overlap with the previous window). 
# Any value of scale that is larger or smaller than one will scale the base image accordingly, 
# resulting in corresponding change in the number of cells per window. 
# Its possible to run this same function multiple times for different scale values to generate multiple-scaled search windows.
    
# for video frame
# Define blocks and steps as above, cell is the basic unit for a step
nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
nfeat_per_block = orient*cell_per_block**2
    
# for classifier images
# 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
window = 64
nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1

# Instead of overlap, define how many cells to step
cells_per_step = 2  
nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
# Compute individual channel HOG features for the region of interest of a image
hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
```

To choose the parameters of the HOG feature extraction, I try to quantify the disimilarity of a car and a not car image with absolute mean difference. To compare the result, I visualized the average difference with boxplot below: 

The following figures shows boxplot of the absolute mean difference of a car image and all not-car images with different HOG parameters
![alt text][image12]

#### Spatial Binning of Color and Histogram of Color

With resized image and color distribution of pixels, we can get features of different pixel resolution and color space. One advantage is to reduce the feature number and keep the relevant features at lower resolution.

A convenient function for scaling down the resolution of an image is OpenCV's `cv2.resize()`. It can be used scale a color image or a single color channel as follows:

```python
# binned color features
color1 = cv2.resize(img[:,:,0], size).ravel()
color2 = cv2.resize(img[:,:,1], size).ravel()
color3 = cv2.resize(img[:,:,2], size).ravel()
binned_color_features = np.hstack((color1, color2, color3))
                        
# the features of the color histogram 
channel1_hist = np.histogram(img[:,:,0], bins=nbins)
channel2_hist = np.histogram(img[:,:,1], bins=nbins)
channel3_hist = np.histogram(img[:,:,2], bins=nbins)
hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
```

The following figures shows the example of a car image (64x64) and its binned color features (16x16)
![alt text][image13]
![alt text][image4]

The following figures shows the line graph of the RGB distribution of a car image
![alt text][image5]

The following figures shows the distribution of a car and a not-car images in different color spaces(RGB, HSV, HLS, LUV, YUV, LAB, YCrCb)
![alt text][image6]

The following figures shows the view of a video frame in different color spaces(RGB, HSV, HLS, LUV, YUV, LAB, YCrCb)
![alt text][image7]

The following figures shows the view of a video frame in RGB color space
![alt text][image8]

### Classifier training
---
To classify the image, I standerdized and shuffled the datasets. Features should be scaled to zero mean and unit variance before training the classifier with `sklearn.preprocessing.StandardScaler()`. Next, I splitted the data into training set(80%) and test set(20%). In the end, I trained a Linear SVM classifer with `scikit learn`. The C parameters of SVM is 1. The parameters of HOG feature extraction are 15 for `orientations`, 8 for `pixels per cell`, 2 for `cells per block`. Number of bin is 32 for the features of histogram of color. Binning size is (32,32). Total feature number is 11988. The test accuracy of SVC model is 0.9918

```python
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

# Use a linear SVC 
svc = LinearSVC()
svc.fit(X_train, y_train)


The following figures shows the boxes with positive predictions from test images
![alt text][image9]

The following figures shows one test video frame with detected boxes, heatmap, `label` image, and the detected boxes based on labelled region.
![alt text][image10]

The following figures shows an example with final detected boxes after thresholding.
![alt text][image11]


Result
1. I tried to compared the Hog features of a car and a not car example based on mean absolute difference. Although the result shows the car image with less features has larger mean abosulte differece over all pixels, the accuracy of classfier did not show better performance
2. I rescaled the window with manually, and it can detect the cars successfully.
3. Relative speed of the car 
4. The size of window is hard-coded. Start from large window, if the large object is detected, the camera cannot see the small vehicle behind the big one.  
5. Collect all the boxes of True positive prediction and calculate the best combination of search bound of each scale.

Conclusion:
1. Different sets of features (HOG, spatial binning of color, histogram of color) could be tested separately based on different color spaces (RGB, HSV, LUV, HLS, YUV, YCrCb, LAB)
2. I expect different window size, large van, hides the sky. It might fail to 
3. Augment data with disappeard low half part, flipping the image
4. Dectec all the lane line and perspective transform and define the window because in real case, we still need to detect lane line. The lane line can be used to estimate the size and aspect ratio of the window.
