#**Traffic Sign Recognition** 
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)
[image1]: ./figures/fig1_plot_of_signs.png "Examples of Traffic Signs"
[image2]: ./figures/fig2_histogram_of_classes.png "Histograms of Labels"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the `numpy.ndarray.shape` and `len(set())` (for counting the number of unique classes) to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

In the following we show 2 examples for each of the 43 sign class in 86 panels. As can be seen from the plot below, the images of the same class can  contain signs of differeing size, shape, bightness, contrast and even sharpness. Some signs even seem to be cropped or hindered partially. 

![Examples of Traffic Signs][image1]

####3. Histograms of the Labels.

Here we plot the histogram for all three sets of data.  It sesms that the training set's distribution more closely resemble that of the test set. For instance, validation set has equal number of cases in clases 20 to 23 while the training and test sets share similar amount in those classes. Overall, the distributions of the three sets of data are fairly similar.

![Histograms of the Labels. Top panel: histogram of class labels in the training set. Middle panel: histogram of class labels in the validation set. Bottom panel: histogram of class labels in the test set.][image2]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

During preprocessing, we processed the image by first converting the image from RGB to YUV, the equations I followed can be found on the [wiki](https://en.wikipedia.org/wiki/YUV#Conversion_to.2Ffrom_RGB):

$Y = 0.299 R + 0.587 G + 0.114 B $

$ U = 0.492 B - Y $

$ V = 0.877 R - Y $

(Note that I made an mistake here that the U channel is actually approximately $0.492(B - Y)$, or, $-0.147R -0.288G + 0.436 B$.  Similar mistake was made with the V channel.  But I have been using this pipeline and can only leave the correct preprocessing into future work.) The Y channel, which is brightness channel, is then histogram equalized to enhance the contrast with `cv2.equalizeHist`.  In [Sermanet and LeCun (2011)](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), they considered similar preprocessing pipeline where
they would first convert image to YUV and then "The  Y  channel is  then preprocessed with  global and local contrast normalization", while U and V channels were left unchanged.

