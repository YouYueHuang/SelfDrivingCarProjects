# Behaviorial Cloning

[//]: # (Image References)
[image1]: ./imgs/center.png
[image2]: ./imgs/flip.png
[image3]: ./imgs/lake_track.png
[image4]: ./imgs/left.png
[image5]: ./imgs/right.png
[image6]: ./imgs/track2_A_LAB.png
[image7]: ./imgs/track2_B_LAB.png
[image8]: ./imgs/track2_A_LAB.png
[image9]: ./imgs/track2_B_LAB.png
[image10]: ./imgs/not_cars.png
[image11]: ./imgs/not_cars.png
[image12]: ./imgs/track2_Cr_YCrCb.png
[image13]: ./imgs/track2_Cb_YCrCb.png
[image14]: ./imgs/not_cars.png
[image15]: ./imgs/track2_H_HLS.png
[image16]: ./imgs/not_cars.png
[image17]: ./imgs/not_cars.png
[image18]: ./imgs/not_cars.png
[image19]: ./imgs/track2_H_HSV.png

<table>
  <tr>
    <td align="center">Autonomous driving around the lakeside track in the simulator</td>
  </tr> 
  <tr>
    <td><a href="https://youtu.be/hGViPj14bw8"><img src='./img/01.gif' style='width: 500px;'></a></td>
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

### File structure
---
The structure and usage of the files in this repository are as follows:

* `model.py`: this script is used to create and train the model.
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
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving. The training images are loaded in **BGR** colorspace using cv2 while `drive.py` load images in **RGB** to predict the steering angles.

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `*.mp4`, so, in this case the video will be `run1.mp4`.

the FPS (frames per second) of the video can be specified. The default FPS is 60.

```sh
python video.py run1 --fps 48
```

### Data collection strategy
---

The training data was collected using Udacity's simulator in training mode on the lakeside track

* The data can be collected by driving it in both counter-clockwise and clockwise direction.
* For gathering recovery data, I also drove along the outer and inner lane line.
* I tried to keep the car in the center of the road as much as possible
* The model need to weave back to the road center if it is away from the road. Without augmentation, the car wobbles noticeably but stays on the road.

* I drove several laps
	(1) two or three laps of center lane driving
	(2) one lap of recovery driving from the sides
	(3) one lap focusing on driving smoothly around curves

Below are example images from the left, center, and right cameras
![alt text][image1]

![alt text][image4]

![alt text][image5]

The following figure is an example of the flipping images
![alt text][image2]

### Preprocessing
#### Color Space
---

Nidia model used YUV color space, and there are also other color space which could recognize the boudary of road and not-road part
(1) Y in YUV
(2) L in LAB
(3) LS in HLS
(4) L in LUV
(5) SV in HSV
(6) RGB

#### Image Cropping
---
The sky part and the front of the car couldn't help predict the steering angle, so I cropped the image by slicing the tensors. We first supply the startY and endY coordinates, followed by the startX and endX coordinates to the slice. That’s it. We’ve cropped the image!

```python
image[60:-25, :, :]
```


#### Image Resizing
---

```python
cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
```



### Model Architecture and Training Strategy
---
The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.



The model predicts only steering angle, not including speed The driving 
The speed of car in autonomous mode could be set in `drive.py` response time


1. My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128, which is similar to the ... . The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer. 
The final model architecture consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture
![alt text][image1]

2. Creation of the Training Set & Training Process with data augmentation

After the collection process, I had X number of data points. I then preprocessed this data by ...

I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.


I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. To combat the overfitting, I modified the model with dropout layers and L2 Regularization in order to reduce overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

3. Model parameter tuning: The model used an adam optimizer, so the learning rate was not tuned manually. I set the initial learning rate 1e-4

4. Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 



5. The overall strategy for deriving a model architecture was to maximize the driving time or reduce the times of being away from the road or dropping down into the lake.

6. I thought this model might be appropriate because of the visualization... 



Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.


    """
    Generate training image give image paths and associated steering angles
    """
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]
            # with augmentation
            if is_training: 
                image, steering_angle = augument(data_dir, center, left, right, steering_angle)
                
            # without augmentation
            else:
                image = load_image(data_dir, center) 
            # add the image and steering angle to the batch
            images[i] = preprocess(image)
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, steers




Nvidia model

```python
"""
NVIDIA model
"""

# Crop the image
model.add(Cropping2D(cropping=((2, 2), (4, 4)),
                     input_shape=(28, 28, 3)))

model = Sequential()
# rescale
model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2))) # kernel_regularizer=regularizers.l2(weight_decay)
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
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(1))
```

I normalized the data with 255 and subtracting 0.5 to shift the mean from 0.5 to 0.





### Result
---
1. The choice of color space. I tried to use different color space. 

1. The data can be collected by driving it in both counter-clockwise and clockwise direction, or the model will perform not well in either direction.

2. The model need to weave back to the road center if it is away from the road. Without augmentation, the car wobbles noticeably but stays on the road.


### Conclusion
---

1. If your model has low mean squared error on the training and validation sets but is driving off the track, this could be because of the data collection process. 

2. The histogram of The data can be collected by driving it in both counter-clockwise and clockwise direction, or the model will perform not well in either direction.






3. The way of collecting datasets would influence the behavior of the model. Clearly classify each recovering action (avoiding hitting the wall, steering back to the road center, etc.) can help detect deficiency of the model and guide the data collecting strtegy. Although the model doesn't need to predict other measurements (brake, speed and throttle), the speed will influence the response time to steer the car.


### Issue
---
* system error [unknown opcode python](https://github.com/keras-team/keras/issues/7297) when running Keras. So I installed a virtual environment of python 3.5.2. 

### Reference
1. [Nvidia self driving car model](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) 