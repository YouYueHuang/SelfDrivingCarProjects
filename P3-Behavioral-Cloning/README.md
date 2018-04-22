# Behaviorial Cloning

[//]: # (Image References)
[image1]: ./imgs/lake_track.png
[image2]: ./imgs/camera_view.png
[image3]: ./imgs/cropping.png
[image4]: ./imgs/resizing.png
[image5]: ./imgs/label_distribution.png
[image6]: ./imgs/steerangle_original.png
[image7]: ./imgs/steerangle_original_moving_average.png

[image8]: ./imgs/track2_A_LAB.png
[image9]: ./imgs/track1_A_LAB.png

[image10]: ./imgs/track1_YUV.png
[image11]: ./imgs/track1_Y_YUV.png
[image12]: ./imgs/track1_U_YUV.png
[image13]: ./imgs/track1_V_YUV.png

[image14]: ./imgs/track1_S_HLS.png
[image15]: ./imgs/track2_S_HLS.png

[image16]: ./imgs/shadowing.png
[image17]: ./imgs/brightness.png

[image18]: ./imgs/model_archi.png
[image19]: ./imgs/loss_linechart.png

<table>
  <tr>
    <td align="center">Autonomous driving in the lakeside track in the simulator</td>
  </tr> 
  <tr>
    <td align="center">The view of Autonomous driving in the lakeside track</td>
  </tr> 
  <tr>
    <td><a href="https://youtu.be/PDS6w3e9rOE"><img src='./img/01.gif' style='width: 500px;'></a></td>
  </tr>
  <tr>
    <td><a href="https://youtu.be/q4AyGOw0ZqQ"><img src='./img/01.gif' style='width: 500px;'></a></td>
  </tr>
</table>

### Overview
---
In this project, deep neural networks and convolutional neural networks were applied to clone driving behavior. A model will be built with Keras to output a steering angle to an autonomous vehicle.

A [simulator](https://github.com/udacity/self-driving-car-sim) from Udacity was used to steer a car around a track for data collection. The image data and steering angles was fed into a neural network model to drive the car autonomously around the track.

### Project goal
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the lakeside track in the simulator. In the meanwhile, the vehicle should remain on the road for an entire loop around the track.

The following figure shows the view of the lakeside track
![alt text][image1]

### File structure
---
The structure and usage of the files in this repository are as follows:

* `model.py`: this script is used to create and train the model. 
* `Main_pipeline.ipynb`: it contain the code for training and saving the convolution neural network. The file show the pipeline I used for training and validating the model, and it contains comments to explain how the code works.
* `drive.py`: this script is used to drive the car.
* `video.py`: this script is used to create a chronological video of the agent driving.
* `model.h5`: a trained Keras model
* `video.mp4`: a video recording of the vehicle driving autonomously around the lakeside track.
* `img`: this folder contains all the frames of the manual driving.
* `driving_log.csv`: each row in this sheet correlates the `img` images with the steering angle, throttle, brake, and speed of the car. The model.h5 was trained with these measuresments to steer the angle.

### Usage
#### Drive the car
---
To run the trained model in the Udacity simulator, first launch the simulator and select "AUTONOMOUS MODE". Then run
the model (model.h5) be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

#### Save a video of the autonomous agent
---

```sh
python drive.py model.h5 video
```

The fourth argument, `video`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving. The training images are loaded in **BGR** colorspace using cv2 while `drive.py` load images in **RGB** to predict the steering angles.

```sh
python video.py video --fps 48
```

Creates a video based on images found in the `video` directory. The name of the video will be the name of the directory followed by `video.mp4`, so, in this case the video will be `video.mp4`. The FPS (frames per second) of the video can be specified. The default FPS is 60.

### Data collection strategy
---

The training data was collected using Udacity's simulator in training mode on the lakeside track

* The data can be collected by driving it in both counter-clockwise and clockwise direction.
* For gathering recovery data, I also drove along the outer and inner lane line.
* I tried to keep the car in the center of the road as much as possible
* I then recorded the car recovering from the left side and right sides of the road back to center so that the vehicle would learn to turn back to the center of the roead

* I drove several laps
	(1) two or three laps of center lane driving
	(2) one lap of recovery driving from the sides
	(3) one lap focusing on driving smoothly around curves

* After the collection process, I had 12840 number of frames from center camera. I then randomly shuffled the data set and put 20% of the data into a validation set. 

Below are example images from the left, center, right cameras and a flipping image
![alt text][image2]

### Preprocessing
#### Color Space
---
Nvidia model used YUV color space, and there are also other color spaces which could recognize the boudary of road and not-road part

The following figure shows the view of 10 frames in (1) YUV color space, (2) Y channel, (3) U channel and (4) V channel respectively

![alt text][image10]

![alt text][image11]

![alt text][image12]

![alt text][image13]

#### Image Cropping
---
The sky part and the front of the car couldn't help predict the steering angle, so I cropped the image by slicing the tensors. I supplied the start and the end of y coordinates to the slice. The code is as follows:

```python
image[start_Y:end_Y, :, :]
```

The following figure shows the frames after cropping
![alt text][image3]

#### Image Resizing
---
I resized the images and made it smaller which can reduce the parameter number of the model. One thing should be noticed is to keep aspect ratio so the image does not look skewed or distorted.

```python
cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
```

The following figure shows the frames after resizing

![alt text][image4]

#### Steering Angle Smoothing with Moving Average and Interpolation
---

When steering the car in training mode, I was just tapping the direction keys for a short moment. That might lead to the significant drop and spike in the measurements of steering angle. Therefore, some frames needs the car to turn direction and it gave an appropriate steering angle, whereas the frames of its previous and following moments got 0 or small steering angle. I applied some moving average and interpolation techniques to compensate those inappropriate measurements.

The following figure shows the histogram of steering angle of (from left to right) (1) center lane driving, (2) recovery driving from the sides and driving around curves, (3) training set, (4) validation set, (5) center lane driving with steering angle smoothing and (6) recovery driving from the sides and driving around curves with steering angle smoothing.

![alt text][image5]

The following figure is the time series plot of steering angle  of (from left to right) (1) before smoothing and (2) after smoothing compensation

![alt text][image6]

![alt text][image7]

### Model Architecture and Training Strategy
---
* The size of image is as follows in each preprocessing stage
	(1) The input size: (160, 320, 3)
	(2) After cropping: (72, 320, 3)
	(3) After resizing: (63, 280, 3)

* Modified Nvidia neural network model

* The following is the parameter of the model
	(1) epoch: 20
	(2) samples per epoch: 19200
	(3) batch size : 64
	(4) learning_rate : 1.0e-4 

* Batch size determines how many examples the model look at before making a weight update. The lower it is, the noisier the training signal is going to be, the higher it is, the longer it will take to compute the gradient for each step.

* The batch of training set was generated with randomly horizontal flipping

* I used this training data for training the model. The validation set helped determine if the model was over or under fitting during 20 epochs. 

* I normalized the data with 255 and subtracting 0.5 to shift the range to -0.5~0.5

* The model used an adam optimizer, so the learning rate was not tuned manually. I set the initial learning rate 1e-4

* My model consists of 5 convolution neural network layers with 3x3 (or 5x5) filter sizes and depths between 24 and 64, which is similar to the [Nvidia model](https://devblogs.nvidia.com/deep-learning-self-driving-cars/). The model includes **ELU** layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. To avoid overfitting and weight exploding, I applied dropout layer with keep probability of **0.2** and batch normalization layer. The total number of parameters is **317,763**, and the evaluation metric is **mean squared error**.

* Due to the limited computation resource, I created the batch on the fly with python `generator`.

Here is a visualization of the architecture
![alt text][image18]

The following is the code of neural network model

```python
"""
Modified NVIDIA model
"""

model = Sequential()
model.add(Lambda(lambda x: x/255-0.5, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)))
model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(50, activation='elu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(10, activation='elu'))

model.add(Dense(1))
```

### Result
---
1. When running the simulator to see how well the car was driving around lakeside track without steering angle smoothing. The vehicle fell off the track at the bridge and the road corners. To improve the driving behavior in these cases, I applied moving average and interpolation techniques to alleviate the abrupt changes of the steering angle. At the end of the process, the car is able to drive autonomously around the track without leaving the road.

2. In order to gauge how well the model was working, I found that the Nvidia model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. To combat the overfitting, I modified the model with dropout layers and smoothing techniques in order to reduce overfitting. The following plot shows both traing loss and validation loss go more consistently.

![alt text][image19]

3. The model predicts only steering angle, not including the driving speed. The speed of car in autonomous mode could be set in `drive.py`, and it could inflence the response time of the agent.

4. With YUV color space, moving average and interpolation, I finally selected model after 19 epoch with mean squared error of 0.182 in both training and validation set. The model of epoch 9 `model-008.h5` could be found in `models`. It can pass the first lap, but stuck in the second lap. The model of epoch 19 can run at least four laps. 

### Conclusion
---

1. From the histogram of steering angle, I found that the distribution is unimodal, and it becomes multimodal after data augmenting for recovery (weaving back to the road center, avoiding hitting the wall, etc.) That means we could change the action and build a robust model by modifying the distribution of the target variable. It also can help compare the pros and cons of different models, and guide the data collecting strtegy. 

![alt text][image5]

2. In mountain track (track 2), there are more shadows and plants on the road, and that might generate noises and lead to bad prediction performance. The model could be trained with artificial noises such as shadow or blot to increase the robustness.

The following figure shows the view of 10 frames with noise of random shadowing

![alt text][image16]

The following figure shows the view of 10 frames with noise of random blotting

![alt text][image17]

3. Although the model doesn't need to predict other measurements (brake, speed and throttle), the speed will influence the response time to steer the car. It is critical to keep the driving speed of traing period and autonomous driving period as close as possible. 

4. Nvidia model used YUV color space, and there are also other color space which could recognize the boudary of road and not-road part listed in the following part. This didn't try to other color spaces, but the model could be improved with different combination of channels from color spaces.

	(1) Y in YUV
	(2) L in LAB
	(3) LS in HLS
	(4) L in LUV
	(5) SV in HSV
	(6) RGB

The following figure shows the view of 10 frames of A channel in LAB color space for (1) lakeside track and (2) mountain track

![alt text][image9]

![alt text][image8]

The following figure shows the view of 10 frames of S channel in HLS color space for (1) lakeside track and (2) mountain track

![alt text][image14]

![alt text][image15]


The following figure shows the view of 10 frames of S channel in HLS color space
![alt text][image1]

### Issue
---
* system error [unknown opcode python](https://github.com/keras-team/keras/issues/7297) when running Keras. So I installed a virtual environment of python 3.5.2. 

### Reference
---
1. [Nvidia self driving car model](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) 
2. [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901)

### Appendix
#### Useful command
---
* Deletes all files and folders contained in the `non_empty_directory_to_be_deleted` directory.
```
rm -rf <non_empty_directory_to_be_deleted>
```

* Counte files in the current directory
```
ls -1 | wc -l
```

* Rename a directory via the command line
```
mv <old_name> <new_name>
```

