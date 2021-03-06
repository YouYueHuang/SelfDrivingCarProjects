## Vehicle Detection

[//]: # (Image References)
[image1]: ./img/cars.png
[image2]: ./img/not_cars.png
[image3]: ./img/HOG_hyperparameter_visualization.png
[image4]: ./img/bin_spatial.png
[image5]: ./img/color_of_histogram.png
[image6]: ./img/histogram_of_color_space.png.png
[image7]: ./img/img_different_color_space.png
[image8]: ./img/color_distribution_of_different_color_space.png
[image9]: ./img/sliding_window.png
[image10]: ./img/heatmap.png.png
[image11]: ./img/output_bboxes.png
[image12]: ./img/HOG_hyperparameter_test.png
[image13]: ./img/image0004.png

<table>
  <tr>
    <td align="center">First half part of vehicle detecting video</td>
    <td align="center">Second half part of vehicle detecting video</td>
  </tr> 
  <tr>
    <td><a href="https://youtu.be/hGViPj14bw8"><img src='./img/01.gif' style='width: 500px;'></a></td>
    <td><a href="https://youtu.be/hGViPj14bw8"><img src='./img/02.gif' style='width: 500px;'></a></td>
  </tr>
</table>


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
The structure and usage of the files in this repository are as follows:

* `Main_pipeline.ipynb`: This part contains the code for feature extraction, preprocessing, classifier training and sliding window.
* `processed_project_video.mp4`: The final result of vehicle detection pipeline.
* `test_images`: It contains images for testing feature extraction, sliding window drawing and.
* `videos`: It contains test videos `project_video.mp4` and the search region of sliding window. 
* `img`: It stores the images of README.md.
* `saved_file`: It contains the `model.p` file. The model stores the parameters for feature extraction.
* `color_space_exploration.py`: The function plots the distribution of the pixels over different color spaces (RGB, HLS, HSV, LAB, LUV, YUV, YCrCb)

### Datasets
---
There are two labeled data to train the classifier
  1. [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip)
  2. [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip).

Total image number: 17760
Number of car images :  8792
Number of non-car images:  8968

<p align="center"><b>The following figures shows the examples of car and not-car images.</b></p>

![alt text][image1]

![alt text][image2]

### Feature Extraction
#### Histogram of Oriented Gradients (HOG)
---
I used the `skimage.feature.hog()` to extracted HOG features just once for the entire region of interest in each full image / video frame. 
Next, I resized the region of interest of a image to solve the car size issue. The size of training samples are 64x64, so we need to resize the samples by sliding window and fit the samples into the classifer. If the prediction is positive, then the window is stored in the candidate list.

<p align="center"><b>The following figures shows the HOG features of a car and a not-car image.</b></p>

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


<p align="center"><b>The following figures shows boxplot of the absolute mean difference of a car image and all not-car images with different HOG parameters.</b></p>

![alt text][image12]

#### Spatial Binning of Color and Histogram of Color
---
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

<p align="center"><b>The following figures shows the example of a car image (64x64) and its binned color features (16x16).</b></p>

![alt text][image13]

![alt text][image4]

<p align="center"><b>The following figures shows the line graph of the RGB distribution of a car image.</b></p>

![alt text][image5]

<p align="center"><b>The following figures shows the distribution of a car and a not-car images in different color spaces(RGB, HSV, HLS, LUV, YUV, LAB, YCrCb).</b></p>

![alt text][image6]


<p align="center"><b>The following figures shows the view of a video frame in different color spaces(RGB, HSV, HLS, LUV, YUV, LAB, YCrCb)</b></p>

![alt text][image7]

<p align="center"><b>The following figures shows the view of a video frame in RGB color space.</b></p>

![alt text][image8]

### Classifier training
---
To classify the image, I standerdized and shuffled the datasets. Features were scaled to zero mean and unit variance before training the classifier with `sklearn.preprocessing.StandardScaler()`. Next, I splitted the data into training set(80%) and test set(20%). In the end, I trained a Linear SVM classifer with `scikit learn`. The C parameters of SVM is 1. The parameters of HOG feature extraction are 15 for `orientations`, 8 for `pixels per cell`, 2 for `cells per block`. Number of bin is 32 for the features of histogram of color. Binning size is (32,32). Total feature number is 11988. The test accuracy of SVC model is 0.9918. 

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

### Sliding Window Search
---
To search vehicles in a frame, sliding window was appied to iterate over predefined interest of region (lower half part of a frame) that could contain cars with boxes of estimated vehicle size, and the classifier will check if a box contains vehicles. As vehicles may be of different sizes due to different perspectives and distortion, boxes need to bed resized for nearby and distant vehicles. I use various sized of sliding window with different scale while iterating with overlapping ratio of 75% in horizontal and vertical directions. I also generated windows with different y starting points and y stop points to reduce search space. In the end, there are false positive patches(not vehicles) and true positive patches(vehicles) for each frame. For false positive detection, thresholding was applied to remove not vehicle part. To locate the vehicles and confirm the number, heatmap and `scipy.ndimage.measurements.label` were used validate the final result in a visual way. The main idea and parameters are extracted as follows:

```python
from scipy.ndimage.measurements import label

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (255,61,0), 4)
    # Return the image
    return img

ystart = [ 400,  400,  410,  420, 430,   430,  440,  400,  400, 500, 400, 400, 440, 400, 410]
ystop = [  500,  500,  500,  556, 556,   556,  556,  556,  556, 656, 556, 556, 500, 500, 500]
scale = [  1.0,  1.3,  1.4,  1.6, 1.8,   2.0,  1.9 , 1.3 , 2.2, 3.0, 2.1, 2.2, 1.0, 1.2, 1.2]
threshold = 1 				# threshold for removing false positive prediction 
color_space = 'YUV' 		# Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 15  				# HOG orientations
pix_per_cell = 8 			# HOG pixels per cell
cell_per_block = 2 			# HOG cells per block
hog_channel = "ALL" 		# Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) 	# Spatial binning dimensions
hist_bins = 32    			# Number of histogram bins

out_img, bboxes = find_cars(image, y_start, y_stop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size)
```

The figures below are the examples of the result of sliding windows, thresholding, heatmap and vehicle labelling.

<p align="center"><b>The following figures shows the boxes with positive predictions from test images.</b></p>

![alt text][image9]

<p align="center"><b>The following figures shows one test video frame with detected boxes, heatmap, `label` image, and the detected boxes based on labelled region.</b></p>

![alt text][image10]

<p align="center"><b>The following figures shows an example with final detected boxes after thresholding.</b></p>

![alt text][image11]


### Result
---
1. I tried to compared the Hog features of a car and a not car example based on mean absolute difference. Although the result shows the car image with less features has larger mean abosulte differece over all pixels, the accuracy of classfier did not show better performance. It is hard to tell if a set of features is benefical to the performance of a classifer. The features still need to be tested collectively, but it will cost considerable computational resources. 

2. I rescaled the window sizes manually, and the results of the detection are acceptible. Sometimes the cars in another side could be detected even though the lower half part of those vehicles are hidden by the road fence. Due to the manual adjustment of parameters, the pipeline will be useless if there is a accidental change of terrain.

3. Sometimes there are a few frames in which the white car is clearly visible and yet it is not detected (false negative detections). One reason for this can be that for certain window sizes the overlap ratio is not big enough, it could be improved by increasing the overlap ratio (i.e. decreasing the value of `cells_per_step`). In the pipeline, I tried to use queue to memorize the position of the vehicle in the previous frames, but it sometimes causes the false positive detection to the vehicles in the oppisite direction. That indicates that this approach is limited to the relative speed of the vehicles and the persistence time of a detected vehicle. 

4. The size of window is square-shaped and hard-coded, but the vehicle such as van or motorcycle might be rectangle. In addition, the vehicle in left and right side might change its shape from different angle. Some parameters such as `cell per block` will increase the search time in square order, so I choose to adjust the search range with different window scale.

### Conclusion
---
1. Although features did not show good capability for prediction in parameter test of HOG feature extraction, different sets of features still can be experimented for classifer training. For example, histogram of color with different color spaces (RGB, HSV, LUV, HLS, YUV, YCrCb, LAB) or combination of different channels might give unique presentations for vehicles.

2. The reduction of sliding window search space is the priority to give a real-time detection. When driving on the road, self-driving car need to detect all the lane lines anyway. With knowing the road condition by lanes, we can apply perspective transform to estimate the aspect ratio and size of window for each region of interest. Also, the lane orientation can indicate the apporximate range of sky which we are not interested in as long as the vehicle is not too large to hide the sky. Another way is to start searching from large window. If the large object is detected, the camera cannot see the small vehicle behind the big one. Therefore, some search with small window size can be skipped. Deep learning techniques without using the sliding windows technique such as [YOLO](https://pjreddie.com/darknet/yolo/) or [SSD](https://arxiv.org/abs/1512.02325) could also be used to detect vehicles with different shapes.

3. The reliability of classifer is important to decrease the occurence of false positive detection because high threshold migh harm the final result of true positive detection. To generalize the classifier and reduce some potential issues such as overfitting, a less costly way is applying the technique of data augmentation. In this project, we can with apply flipping, slight rotation (in case of a turn of terrain change) and add some noises to the low half part of a vehicle (in case of being hidden by obstacles partialy.)

4. There are some false positive results with horizontal lines. To improve the performance of the classifier, we could adopt incremental learning with `partial_fit` in `sklearn.linear_model.SGDClassifier` and the linear SVM could be used with loss function of `hinge`. When removing the false positive patches with thresholding techniques, we can recycle and feed them into the model to help classifer recognize them next time. There are other methods, such as requiring that a detection be found at or near the same position in several subsequent frames, (could be a heat map showing the location of repeat detections) is implemented as a means of rejecting false positives, and this demonstrably reduces the number of false positives. Same or similar method used to draw bounding boxes (or circles, cubes, etc.) around high-confidence detections where multiple overlapping detections occur.

5. To get even better classification results, fine tune the C parameter of LinearSVC classifier could gives a good controls between the training accuracy and generalization. As the test and training images came from the same source, high test accuracy might imply overfitting. Against overfitting you need more generalization, meaning C<1.0 values such as 0.01 or even 0.0001. Also, the result of the method `decision_function()` of LinearSVC can be thresholded to avoid false positives.

### Reference
---
1. [SVM with inncremental learning in Scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier)
2. [Real time object detection with YOLO](https://pjreddie.com/darknet/yolo/)
3. []
4. []