# **Traffic Sign Recognition** 

## Writeup


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/figure_1.png "Visualization"
[image2]: ./examples/figure_1-1.png "Visualization"
[image3]: ./examples/figure_1-2.png "Normalization"
[image4]: ../data/3.jpg "Traffic Sign 1"
[image5]: ../data/11.jpg "Traffic Sign 2"
[image6]: ../data/18.jpg  "Traffic Sign 3"
[image7]: ../data/25.jpg "Traffic Sign 4"
[image8]: ../data/34.jpg  "Traffic Sign 5"
[image9]: ./examples/figure_1-3.png  "Prediction on 5 imgaes"
[image10]: ./examples/figure_1-4.png  "Prediction on 5 imgaes"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
Here is a link to my [project code](./Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Basic summary of the data set.
I used python to calculate summary statistics of the traffic
signs data set:

* The size of training set is 39209
* The size of the validation set is 7842
* The size of test set is  12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an example of the data set images. It is amazing how convnets are able to be trained on these images
and delivery high recognition accuracy.

![alt text][image1]

Histogram plotting of the original training data set shows that some classes may not have enough data required for high 
accuracy recognition. Secondly, distribution of the test data on the other side is quite similar to the train data, so I
 would expect that sign recognition will not be biased to a particular sign.
 
![alt text][image2]

### Design and Test a Model Architecture

#### 1. Image preprocessing

To improve the accuracy the first thing I did I increased the data set size. Initially, I implemented several steps: 
rotation, warping, shifting the images in x and y coordinates. But the data size increased hugely so that my pc would 
run out of memory very quickly. At the end, I just keept only image rotation option and the data set increased by 5.

After some trying I kept the normalization as it performed slightly better than standardization:

![alt text][image3]

Other things I tried shortly were playing with color spaces (YUV for example as mentioned in  
[[LeCun]](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)
 and gray colorspaces) and histogram equalization. But haven't seen much of a progress.


#### 2. Model architecture

My final model consisted of 2 convolutional layers and 2 fully connected layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x64  				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x128  |   
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x128  				|
| Fully connected		| Input = 3200, output = 64          			|
| RELU					|												|
| Fully connected		| Input = 64, output = 43           			|
| Softmax				|           									|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model I used Adam optimizer. Some advantages of Adam include:

* Relatively low memory requirements
* Usually works well even with a little tuning of hyperparameters

Batch size was set to 128 due to memory constraint. To avoid overfitting data augmentation was used. 
The number of epoch was set to 10 (as I wanted to save GPU time for later projects), reaching 96% of accuracy.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.999 
* test set accuracy of 0.956

With LeNet architecture I reached 94% accuracy. After that I slightly modified it, I removed one of the fully connected
 layers and increased the depth of the activation volume instead. The idea here was to increase the depth column in 
 order to get more details as mentioned in [[LeCun]](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) 
 "In the case of 2 stages of features, the second stage extracts
“global” and invariant shapes and structures, while the first
stage extracts “local” motifs with more precise details." 


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The images are actually easily recognizable, especially when compared to the traing data set, so I don't expect to have any problems here,
the only thing perhaps they are a bit too bright and too clean and may confuse the model.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).
The accuracy is 100% which is better than training accuracy (99.9%) and validation accuracy (99.9%). But the number of 
images is of course not enough for the statistics and will change as we introduce more and more images.
#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

![alt text][image9]

The model is 100% certain for all 5 images. Which as I mentioned is likely to change as soon as we introduce more images.

![alt text][image10]


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


