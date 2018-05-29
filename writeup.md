# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Training.png "Model MSE Loss"
[image2]: ./examples/center.jpg "Centre"
[image3]: ./examples/left.jpg "Left"
[image4]: ./examples/right.jpg "Right"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with various filter sizes and depths between 24 and 64 (model.py lines 57-73). It is based upon the Nvidia model, but with a few extra layers.

The model includes RELU on the five convolution layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 59). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 67, 69, 71). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track, and also drive smoothly.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 75).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and also driving the second test track.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to find a known working model and adapt and optimise it to fit the problem. 

The model developed by Nvidia (https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) was used as a base, and extra convolution layers and parameters were added to account for the different input image size and dataset. The Nvidia model has an input size of 200x60x3 and 250k parameters, my model has an input size of 320x80x3 and 348k parameters.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a high MSE on both the training and validation sets, which implies underfitting. I adjusted the subsampling on the convolution layers, and also added extra convolution layers, so that the output size from the convolution layers in my model was aproximately the same as the nvidia model.

This helped dramatically, but the car was still unable to navigate some of the sharp corners correctly. I created a new set of input images with the simulator in both the forward and reverse direction, including a number of recoveries from the edge back to the middle. This fixed the problem.

I has originally set the number of epochs to 5, but after experimenting with dropout layers I increased this to 15 as there was no overfitting, and that seems to be the point of diminishing returns.

![alt text][image1]

I also discovered that the Dense(1164) layer in the Nvidia paper was not needed for the simple track, so I commented it out. Later I found that training with more epochs made the model good enough for track 2 as well. Training on each epoch takes 58s with a GTX 1080 ti.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
cropping2d_1 (Cropping2D)        (None, 80, 320, 3)    0           cropping2d_input_1[0][0]
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 80, 320, 3)    0           cropping2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 38, 106, 24)   1824        lambda_1[0][0]
(5x5 kernel, 2x3 subsampling)
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 17, 34, 36)    21636       convolution2d_1[0][0]
(5x5 kernel, 2x3 subsampling)
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 7, 15, 48)     43248       convolution2d_2[0][0]
(5x5 kernel, 2x2 subsampling)
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 5, 13, 64)     27712       convolution2d_3[0][0]
(3x3 kernel)
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 3, 11, 64)     36928       convolution2d_4[0][0]
(3x3 kernel)
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2112)          0           convolution2d_5[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 2112)          0           flatten_1[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           211300      dropout_1[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 100)           0           dense_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dropout_2[0][0]
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 50)            0           dense_2[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dropout_3[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]
====================================================================================================
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0
```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving, in the forwards and reverse direction.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to self correct if it gets too close to the edge of the road

Then I repeated this process on track two in order to get more data points, this time in the forwards direction only.

To augment the data set I also used images from the left and right cameras (with a small steering angle correction applied to account for the different perspective), and a mirrored version of the first three. For example, are images from the Left, Center, and Right cameras:

![alt text][image2]
![alt text][image3]
![alt text][image4]

After the collection process I had 37,692 images, which when argumented as above gives 226,152 data points. The model has built in normalisation of the data

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.
