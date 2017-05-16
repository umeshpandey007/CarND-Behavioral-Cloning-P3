# **Behavioral Cloning Project** 

---

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/image1.png "Model Visualization(Ref: Modified version of Nvidia's self-driving car model)"
[image2]: ./examples/image2.jpg "Center lane driving"
[image3]: ./examples/image3.jpg "Recovery Image 1"
[image4]: ./examples/image4.jpg "Recovery Image 2"
[image5]: ./examples/image5.jpg "Recovery Image 3"
[image6]: ./examples/image6.jpg "Recovery Image 4"
[image7]: ./examples/image7.jpg "Recovery Image 5"
[image8]: ./examples/image8.jpg "Recovery Image 6"
[image9]: ./examples/image9.jpg "Normal Image"
[image10]: ./examples/image10.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with filter sizes varying from 5x5 to 3x3 and depths of 12,18,24,36,48 (model.py lines 95-125) 

The model includes RELU layers to introduce nonlinearity (code line 102,108,117,123,129), and the data is normalized in the model using a Keras lambda layer (code line 90).

#### 2. Attempts to reduce overfitting in the model

Initially tried dropout layers to prevent overfitting but was getting marginal improvements, instead training on more data helped in generalization. I had collected my own set of driving data and combined it with sample data given by Udacity. This helped in generalization of the driving behaviour. Roughly the model was trained on 40k images (20k images captured from training simulation + 20k flipped version of the images captured from training simulation)

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 149). 

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. Also I combined the data from udacity and training data captured by me.The training data also consisted of one lap where in the vehicle was driven in the clockwise direction.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to reuse an existing model and tune to get better performance on this problem

My first step was to use a convolution neural network model similar to the [Nvidia CNN model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). I thought this model might be appropriate because it was already proved to be working, however I did not used exacly same model instead simplified and tried only few layers from the Nvidia CNN model.

To combat the overfitting, I initially had added dropout layers but it gave marginal improvement to validation loss. So, I decided to exclude the dropout layers and instead combined the training data from both udacity and the training data collected by me.

There were multiple iterations required to get a good model, I experimented with multiple number of feature maps and filter sizes in CNN to tune the model. Once I got a validation loss of 0.0063, I stopped tuning the model as I felt it was good enough to go for testing the vehicle in autonomous mode.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track especially on steep turns. To improve the driving behavior in these cases, I collected few more training data near the turns, especially the turns where the vehicle went off-road.

In the process of data collection, I realized that the data collected by navigating with the keyboard gave very poor results as there was always a sharp step change in the steering angles and the vehicle was emulating this behaviour which resulted it in going off-track. So, I decided to collect the training data by using mouse to get continuous steerting angles, which helped in overall better data.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 76-152) consisted of a convolution neural network with the following layers and layer sizes:

Here is a visualization of the architecture:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![Center Lane Driving][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover if it starts driving offroad. These series of images show what a recovery looks like:

![Recovery Lane Driving 1][image3]
![Recovery Lane Driving 2][image4]
![Recovery Lane Driving 3][image5]
![Recovery Lane Driving 4][image6]
![Recovery Lane Driving 5][image7]
![Recovery Lane Driving 6][image8]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking so that the data is not biased to only left turns For example, here is an image that has then been flipped:

![Normal Image][image9]
![Flipped Image][image10]


After the collection process, I had around 40K number of data points. Data preprocessing was added as part of model by adding normalization function with Lambda layer. Also, I added a Cropping layer in the model with 60 pixels cropped from the top and 25 pixels from the bottom, so that the model does not learns of extraneous features except of the tracks.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by validatin loss of 0.0063(model.py line 150).  I used an adam optimizer so that manually training the learning rate wasn't necessary.
