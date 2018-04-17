# Behaviorial Cloning

[//]: # (Image References)
[image1]: ./img/cars.png
[image2]: ./img/not_cars.png
[image3]: ./img/HOG_hyperparameter_visualization.png
[image4]: ./img/bin_spatial.png

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]

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

### Drive the car
---
the model (model.h5) be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing **,** with **.** when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add `export LANG=en_US.utf8` to the bashrc file.

### Save a video of the autonomous agent
---

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
...
```

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
1. The data can be collected by driving it in both counter-clockwise and clockwise direction, or the model will perform not well in either direction.
2. The model need to weave back to the road center if it is away from the road. Without augmentation, the car wobbles noticeably but stays on the road.

3. Drive in the opposite direction
4. Mirror
5. Use different tracks
6. Drive two or three laps of center lane driving
7. One lap of recovery driving from the sides
8. One lap focusing on driving smoothly around curves

### Model Architecture
---

```python
# Clip the image
model.add(Cropping2D(cropping=((2, 2), (4, 4)),
                     input_shape=(28, 28, 3)))
# now model.output_shape == (None, 24, 20, 3)
```

I normalized the data with 255 and subtracting 0.5 to shift the mean from 0.5 to 0.

### Result
---

### Conclusion
---

If your model has low mean squared error on the training and validation sets but is driving off the track, this could be because of the data collection process. 
