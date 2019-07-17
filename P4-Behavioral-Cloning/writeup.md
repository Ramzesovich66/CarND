# **Behavioral Cloning** 



**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior or use already prestored driving data
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia.png "Model Visualization"
[image2]: ./examples/hist.PNG 
[image3]: ./examples/center_2016_12_01_13_30_48_287.jpg "Recovery Image"
[image4]: ./examples/left_2016_12_01_13_30_48_287.jpg "Recovery Image"
[image5]: ./examples/right_2016_12_01_13_30_48_287.jpg "Recovery Image"
[image6]: ./examples/normal.png "Normal Image"
[image7]: ./examples/flipped.png "Flipped Image"

## Rubric Points

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.json containing the architecture of a model
* model.hdf5 containing weights of the CNN
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py 
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
I slightly modified (model.py lines 14 - 43) [Nvidia CNN architecture](https://arxiv.org/pdf/1604.07316v1.pdf) in which they 
try to solve the same problem of steering angle prediction.
The network consists of 9 layers, including a normalization layer, 5 convolutional layers, 3 fully connected layers, finally, 
a last single neuron tries to regress the correct steering value from the features it receives from the previous layers:

![alt text][image1]

In my case, the input layer I ended up with is 64x64x3. The first layer of the network performs image normalization. 
The normalizer is hard-coded and is not adjusted in the learning process. It is implemented through a `Lambda` layer 
such the input is in the range [-0.5, 0.5].
The model includes ELU layers to introduce nonlinearity and has the following advantages over RELU: 
* ELU becomes smooth slowly until its output equal to -α whereas RELU sharply smoothes,
* unlike to ReLU, ELU can produce negative outputs.

Additionally, before using an image for a training, I croped the image to remove sky, forest and the hood of the car and then 
resized it to a smaller image.

#### 2. Attempts to reduce overfitting in the model

To prevent overfitting, dropout layers are added after each convolutional layer (prob = 0.2) 
and after the first three fully-connected layer (prob = 0.5).

Additionally, data augmentation was performed to ensure that the model was not overfitting (prepare_data.py line 31 - 52). 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer (model.py lines 64):
 * no need to tune the learning rate manually,
 * computationally efficient,
 * suited for problems that are large in terms of data and/or parameters

### Data set and visualization

The training data can be generated with Udacity simulator where the user can control the car through the keyboard and 
the mouse, and store frames with corresponding steering angles. Udacity also provided explicitly a training set 
with around 8000 images ( from the frontal, left and right cameras) that I eventually used for the project:

![alt text][image3]
![alt text][image4]
![alt text][image5]

Steering angle distribution is strongly biased towards zero: 

![alt text][image2]

### Data augmentation

Due to relatively small number of training data (and its bias towards zero steering angle) data augmentation was performed.
Firstly, images from side cameras were used by appropriately correcting the steering angle, i.e. steering angle of left images 
were incremented by 0.2 and decremented by 0.2 for the right camera images. Secondly, images were randomly flipped with
 their steering angle changing the sign: 
 
![alt text][image6]
![alt text][image7]

 While these two ways of augmenting the data were enough for successfully 
 driving the car autonomously, additionaly, images were shifted horizontally and vertically 
 (borrowed [here](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9)) to simulate different
 position of the car on the road, appropriately correcting the steering angle (adding/substracting 0.004 steering angle
  units per pixel shift) and allowing to perform autonomous driving with the best graphics settings ==> [video](run1.mp4)

