
# Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### My project includes the following files:

File | Description
------------ | -------------
<a href="https://github.com/TheOnceAndFutureSmalltalker/street_sign_recognition/blob/master/model.py">model.py</a> | Python code defining the neural network used to train the model from input images and throttle angles
model.h5 | The model developed by the neural network that mimics human driving of the simulated car
<a href="https://github.com/TheOnceAndFutureSmalltalker/behavioral_cloning/blob/master/drive.py">drive.py</a> | Python code used to load and run the model in the car simulator
<a href="https://github.com/TheOnceAndFutureSmalltalker/street_sign_recognition/blob/master/writeup.md">writeup_report.md</a> | Full description of the project and suggestions for improvement
<a href="https://github.com/TheOnceAndFutureSmalltalker/behavioral_cloning/blob/master/video.mp4">video.mp4</a> | Video of AI program driving the car around the track

#### Code Function and Execution
1. Driving data is first acquired by executing the simulator in <e>record mode</e> and saving images and a log file of numeric data to a folder.
```sh
windows_sim.exe
```

2. The images and log file are then read in by model.py which then uses the input data to train a convolutional neural network to drive the simulator and saves the model to the file model.h5.
```sh
python model.py
```

3. The drive.py file is then used to test the trained model, again using the simulator, by providing the model's suggested steering angle given the current road image of the simulator. 
```sh
python drive.py model.h5
```

4. If the model performs well, a video of its driving can be captured in a folder using video.py.
```sh
python drive.py model.h5 <folder>
```

#### Submission Code Readability and Usability

The model.py file contains the Python/Keras code for training and saving the convolution neural network. It loads the images and steering angles from the simulation output.  It does any image manipulation necessary.  It defines the neural network architecture.  It defines the cost function and optimization function.  It defines any hyper paranmeters required for training.  It then trains and saves the model to a file.

The drive.py file is the same as that provided by the Udacity project download from github and is unchanged.


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I initially tried an AlexNet architecture but did not get adequate results.  I then switched to the NVIDIA architecture which was developed specifically for self driving cars.  I tweaked this a bit resulting in the following 8 layer architecture. 

Layer | Description
------------ | -------------
Layer 1 | Input layer of image dimension 160 X 300 X 3, normalized, then cropped to output dimension 70 X 320 X 3
Layer 2 | 5 X 5 convolutional layer with 2 X 2 pooling, relu activation, resulting in output dimension 33 X 158 X 24 
Layer 3 | 5 X 5 convolutional layer with 2 X 2 pooling, relu activation resulting in output dimension 15 X 77 X 36
Layer 4 | 5 X 5 convolutional layer with 2 X 2 pooling, relu activation resulting in output dimension 6 X 37 X 48
Layer 5 | 3 X 3 convolutional layer, relu activation resulting in output dimension 4 X 37 X 64
Layer 6 | 3 X 3 convolutional layer, relu activation resulting in output dimension 2 X 37 X 64
Layer 7 | Flatten to output dimension 4736
Layer 8 | Fully connected layer with output dimension 100 with Dropout rate of 0.5
Layer 9 | Fully connected layer with output dimension 50 with dropout rate of 0.5
Layer 10 | Final output of dimension 1


#### 2. Attempts to reduce overfitting in the model

To prevent overfitting I used the following strategies.  

1. Stop training once the training and validation losses stopped reducing.  This happened very quickly for this architecture.  Just 2 or 3 epochs was usually enough.  Especially for larger data sets which run several more mini batches.  

2. Increased the data set making it less likely the model will overfit.

3. Employed dropout on the last 2 fully connected layers (model.py lines 50 & 52)

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 57).

The number of epochs was eventually tuned to 3 (model.py line 60).  I tried 1, 2 and aas high as 20.  The lowest errors seemed to occur by epoch 3 without having any benefit to training further.

The dropout rate was 0.5 and not tuned at all.

I ended up using a training/validation split of 0.9/0.1 because the data set I eventually used had over 8,000 images and I felt 10% of that was fine for validation giving me more images to train on.

The raw images were cropped to eliminate those parts of the image better focus the training on those parts of the image that actually contain the road.  For this purpose, the images were cropped 70 pixels from top and 20 pixels from bottom resulting in mage dimension of 90 X 320 X 3.  Nothing was cropped from the sides.

#### 4. Appropriate training data

After several failed attempts at generating my own training data, I ended up using the training data supplied by Udacity.  It was much higher quality and more data than I could provide.  

For details of my own experiences capturing data, see the section below.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
