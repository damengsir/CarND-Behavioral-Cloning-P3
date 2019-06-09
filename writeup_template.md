# **Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image2]: ./IMG/center_drive.jpg "Center driving Image"
[image3]: ./IMG/recovery_1.jpg "Recovery Image"
[image4]: ./IMG/recovery_2.jpg "Recovery Image"
[image5]: ./IMG/recovery_3.jpg "Recovery Image"
[image6]: ./IMG/original.jpg "Normal Image"
[image7]: ./IMG/flip.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  
---
### Files Submitted & Code Quality
#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
First, I normalized the data using a Keras lambda layer (code line 62)

Second, I removed the useless features from the pictures.(line 63)

Third, I used 5 convolutional layers to extract features, and used RELU function to introduce nonlinearity(line 64~68)

Forth, I used 4 fully connected layers.(line 69,70,72,73)

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 71). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 15). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 76).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach
First, I used one convolutional layer and two fully connected layers to make up a model to get familiar with the whole process.

In order to got more data to train my model, I used the left, right, center camera's images, and do a data enhancement with a horizontal flip.

Then, I used the network architecture which comes from NVIDA. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my second model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model with Dropout, I found that different places and different numbers of Dropout layers have different effects to the final result. 

I also adjust the angle which will be used to calculate the steer angle from left and right cameras. Finally, I thought 1.4 is the best value. But When there is a big turning, it always out of the road. I can't find a value to adapt the road.

Then I only used the center camera's images. Amazing, the car runned very well on the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 61-74) consisted of a convolution neural network with the following layers and layer sizes.

|Layer (type)      | Output Shape |
|--|--|
| lambda_1 (Lambda) | (None, 160, 320, 3) |
|cropping2d_1 (Cropping2D)| (None, 65, 320, 3)|
|conv2d_1 (Conv2D) | (None, 31, 158, 24) |
|conv2d_2 (Conv2D)|(None, 14, 77, 36)|
| conv2d_3 (Conv2D)| (None, 5, 37, 48) |
| conv2d_4 (Conv2D) | (None, 3, 35, 64)|
| conv2d_5 (Conv2D) | (None, 1, 33, 64)|
|flatten_1 (Flatten)|    (None, 2112) | 
| dense_1 (Dense) |    (None, 100)|  
|dropout_1 (Dropout)| (None, 100) |
|dense_2 (Dense)| (None, 50)|
| dense_3 (Dense) |  (None, 10)|
| dense_4 (Dense) | (None, 1)|

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer back to the center of the road. These images show what a recovery looks like starting from the right side to the road's center:

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the data sat, I also flipped images and angles thinking that this is a simple way to extend the data set. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]


After the collection process, I had 4964 number of data points. 


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4. Because when the epochs is 4, the mean squared error is the smallest. I used an adam optimizer so that manually training the learning rate wasn't necessary.
