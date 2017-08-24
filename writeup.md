#**Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./center.jpg "Center Aligned"
[image3]: ./recover_left_center.jpg "Recovery Image"
[image4]: ./recover_left_center2.jpg "Recovery Image"
[image5]: ./recover_left_center3.jpg "Recovery Image"
[image6]: ./original.jpg "Normal Image"
[image7]: ./flipped.jpg "Flipped Image"
[image8]: ./drive.jpg "Running Model"
[image9]: ./trainingepochs.jpg "Training/Validation Errors with Epochs"

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a 5 convolution neural network with different filter sizes varying from 3x3 filter sizes to 5*5 and depths between 24 and 64 (model.py lines 65-85). The model includes RELU layers to introduce nonlinearity (code line 71), the activation layers are added with the convolution layers. Following this I added 4 direct fully connected layers with depts 100, 50, 10 and 1 respectively. 

The model was fed preprocessed data, in the preprocessing layer I normalize, mean center and crop the data. 
To visualize the loss, I am using the generator with model.fit_generator() from keras, which outputs the progress bar, loss metric and the loss on training and validation sets after each epoch. This graph helps a lot in determining the overfitting/underfitting.

Also since huge amount of data is used to train the model, it cant be fit all together in memory. Generators allow us to divide the dataset and load the data in an iterative manner such that memory of the system is not overloaded. Instead of storing pre processed data in memory all at once, using a generator we can pull pieces of data and process them on the fly when we need them, which is much more memory efficient.
Uses the yield functionality which stores the current iteration values and when it is called second time, proceeds further from that point.

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 80,75) with probablity .2. The number of dropout layers and the drop probablity was decided on hit and trial basis. 

![alt text][image9]

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 13). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 85).
Number of epochs was determined by running the model multiple times and checking the underfit/overfit criterias.
Drop rate - .2 hit and trial and 2 layers were added 2 after CNN and 1 after fully connected layer.

![alt text][image8]


####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture is based on NVIDIA's "End to End Learning for Self-Driving Cars".
Source:  https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

My first step was to use a convolution neural network model similar to the LeNet but it didnt work quiet well and then from the video lectures got the idea of using NVIDIAs architecture. This approach works good because it has so many convolution layers with different depths, hence capturing all the different features like
- track
- Texture of the road
- Water body - it needs to understand and stay away from water, this feature is important
- Sand part 
- Trees
- track boundaries and fences
- sharp turns

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. Initially my model had low validation and low training error and was not working well. It was underfit and then I included more data corresponding to sharp turns and different places. Then I found that my model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to include drop outs.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track but with addition of more and more training data, I was able to get it working.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 64-84) consisted of a convolution neural network with the following layers and layer sizes ...

Input
1. Preprocessing Layer (Lambda - x/255 -.5) for input shape (160,320,3)
2. Cropping (70,25) (0,0) - 70 from top and 25 from bottom
3. 5 Layers of Convolution Networks
- 5*5 filter with 24 depth and Activation RELU
- 5*5 filter with 36 depth and Activation RELU
- 5*5 filter with 48 depth and Activation RELU
- 3*3 filter with 64 depth and Activation RELU
- DropOut layer .3 drop probability
- 3*3 filter with 64 depth and Activation RELU

4. DropOut Layer (.2) drop probability
5. Flatten
6. 4 Layers of Fully Connected networks
- Dense(100) 
- Dropout (.2)
- Dense(50)
- Dense(10)
- Dense(1)

Adam optimizer is used so learning rate issue is addressed by the model itself.


####3. Creation of the Training Set & Training Process

Data Statistics
Data consists of around total 16000 images. The images are taken from 3 cameras mounted on the car, center, left and right. At each point 3 images are capured, this helps in better training process as it provides different viewing angles and views and hence reboust data set. Also I used Augmentation to generate more data. With this project I realized collection of proper dataset is one of the most important task as I had to go over 5-6 iterations to get proper data set. The car would cover some parts of track beautifully but would screw up when there is change of scenario. Its important to realize the different landscapes present in the data set
1. Water
2. Desert 
3. Bridge
4. Normal RoadPath
5. Little bit of hills around. 
Data gathering was required for each specific part, then only model could properly.

Training was one of challenging tasks of the project. We were to take care of not only covering all the aspects (angles at which it turns, driving in the middle of the track) but also take care of recovering the car if it deviates from the track. My data collection step was as followed:

1. To capture good driving behavior, I first recorded two laps on track one using center lane driving. The code in model.py lines 26-29 shows that part. Here is an example image of center lane driving:

![alt text][image2]

2. I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to come back on track in case it goes outside. This was also needed so as to take care of sharp curves. In the example video there were various points of sharp curves, without this training set, car would loose the track on extreme curves and go on sideways. This was to done at multiple places in the training, extreme sharp corners, and little steep corners, they had different behaviour. 
The code in model.py lines 39-49 represents this.
These images show what a recovery looks like starting from  left to right:

![alt text][image3]

![alt text][image4]

![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles (model.py lines 34-38). This process helps in couple of ways. 
1. More training data and more robustness
2. Data becomes much more comprehensive
3. This is an effective technique helping with left turn bias. Another approach which I didnt try was to shift the images horizontally/vertically. 

![alt text][image6]
![alt text][image7]


After the collection process, I had around 16000 number of data points. I then preprocessed the data:
1. Normalization: Normalized the data using Lambda to bring it to the scale (0-1). 
2. Mean Centered the image by subtracting .5.
This reduced the training/validation loss.

Training and Validation Set
I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. If training error decreases but validation error increases, it is definitely a sign that we are overfitting the model, we should reduce the #epochs in that case. The ideal number of epochs was 7 as evidenced by the training/validation error.

In the model, I also am cropping the images, because top portion of the image captures trees, sky while the bottom portion captures hood of the car. Our model does not need those feautures to drive the car on track, so to get rid of those features, I Cropped the image from top and bottom. This indeed improves the results to some extent.

