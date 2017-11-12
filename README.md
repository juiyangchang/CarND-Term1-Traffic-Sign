# **Traffic Sign Recognition** 
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
[image1]: ./figures/fig1_plot_of_signs.png "Examples of Traffic Signs."
[image2]: ./figures/fig2_histogram_of_classes.png "Histograms of Labels."
[image3]: ./figures/fig3_preprocess.png  "Example of Preprocessed Image."
[image4]: ./figures/fig4_dataaugmentation.png "Examples of Transformed Data"
[image_pca]: ./figures/alexnet_rgb_pca.png "AlexNet Fancy PCA"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.*

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

**Data Set Summary & Exploration**

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I use the `numpy.ndarray.shape` and `len(set())` (for counting the number of unique classes) to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

In the following we show 2 examples for each of the 43 sign class in 86 panels. As can be seen from the plot below, the images of the same class can  contain signs of differeing size, shape, bightness, contrast and even sharpness. Some signs even seem to be cropped or hindered partially. 

![Examples of Traffic Signs][image1]

#### 3. Histograms of the Labels.

Here we plot the histogram for all three sets of data.  It sesms that the training set's distribution more closely resemble that of the test set. For instance, validation set has equal number of cases in clases 20 to 23 while the training and test sets share similar amount in those classes. Overall, the distributions of the three sets of data are fairly similar.

![Histograms of the Labels. Top panel: histogram of class labels in the training set. Middle panel: histogram of class labels in the validation set. Bottom panel: histogram of class labels in the test set.][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

##### Preprocessing

During data preprocessing, we process the image by first converting the image from RGB to YUV, the equations I followed can be found on the [wiki](https://en.wikipedia.org/wiki/YUV#Conversion_to.2Ffrom_RGB):

* Y = 0.299 R + 0.587 G + 0.114 B 
* U = 0.492 B - Y 
* V = 0.877 R - Y 

(Note that I made an mistake here that the U channel actually approximately 0.492(B - Y), or, -0.147R -0.288G + 0.436 B.  Similar mistake was made with the V channel.  But I have been using this pipeline and can only leave the correct preprocessing into future work.) The Y channel, which is brightness channel, are then histogram equalized to enhance the contrast with `cv2.equalizeHist`.  In [Sermanet and LeCun (2011)](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), they considered similar preprocessing pipeline where they would first convert image to YUV and then "The  Y  channel is  then preprocessed with  global and local contrast normalization" (quote from their paper), while U and V channels were left unchanged. Finaly we normalize each channel by `X = (X - 128) / 128.`, this will make each channel be within the range of [-1, 1].

Below I show an example of the preprocessed image.
![Example of preprocessed image.][image3]

Another preprocessing step we will do is to one-hot encode the y-labels.

##### Data Augmentation

While our training set data is of moderate size, chances are that we can still be short of data when we try going deep. As seen from previous plots, real world images can be taken from different standing points, viewing angles and during a different day time or weather. In the following, I will use the `ImageDataGenerator` API provided by [keras](https://keras.io/preprocessing/image/) to perform random linear transformations (shifting, rotating) to the training images.  In addtion, I will follow the PCA proecdure described in the [AlexNet paper](https://www.cs.toronto.edu/~kriz/imagenet_classification_with_deep_convolutional.pdf) to purturb image intensities in RGB channel.

##### *Linear Transformation with keras.preprocessing.image.ImageDataGenerator*

The `ImageDataGenerator` is an API defined in `keras.preprocessing.image.py` and can be imported with the following line:
```python
from keras.preprocessing.image import ImageDataGenerator
```
To begin with I create a `ImageDataGenerator` object `datagen` with the following transformation types and ranges: rotate by [-15, 15] degrees, vertically and horizontally shift by [-2, 2] pixels, and shear by [-10, 10] degrees.

The first two types of transformations were used in the traffic sign recognition paper ([Sermanet and LeCun, 2011](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)).  They also did image scaling in the paper but
```ImageDataGenerator``` would scale (by setting ```zoom_range``` to a number or a list) along height and width with two different scaling ratios, thus unnaturally deform the sign shape.  Image scaling can be done manually with a transformation matrix or with opencv3 library but I didn't consider going in that direction.  The transforamtion
ranges are hyperparameters potentially worth tuning but I didn't explore the direction.

Sometimes datagen would be fitted to training set (`datagen.fit(X_train)`) prior to applying the trasnformation but since our transformation doesn't depend on the population statistics of the training data we don't need to do it here. 

During trainging we can run something like the following to train for over training data with
an augmentation factor of 10.
```python
counter, batches = 0, math.ceil(10 * X_trian.shape[0] / 64)   # augment the data by a factor of 10
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=64):
    # do the training
    ...
    counter += 1
    if counter == batches:
        break
    
```
It will random shuffle the data and apply all types of transformations to each image. The ```datagen.flow()``` is a generator that will run indefinitely, which is why we have to stop the loop by checking number of batches trained over.

##### ***Color-channel Purturbation with PCA***

Another transformation we will consider is to purturb the RGB channels by adding noises. We follow the procedure mentioned in the [AlexNet paper](https://www.cs.toronto.edu/~kriz/imagenet_classification_with_deep_convolutional.pdf).

To begin with we compute the covariance matrix of the RGB channels. Next we compute the priciple compontents `lam` and their feature directions P with eigendecompostion.  Following the AlexNet paper, during training we add noise to each of RGB channels according to

![][image_pca]

where p_i is the eigenvector of the covariance matrix and lambda_i is the square root of the eigenvalues  while alpha_i's are a gaussain random variable drawn for each image. Unlike AlexNet, we use
the lambda_i as square root of the eigenvalues as this would ensure the added noise would have the same covariance matrix as the image when alpha_i's are standard normal.

In AlexNet paper, alpha_i's are of standard deviation 0.1, here I use standard deviation of 0.05, which, as shown in the following plot, in the extreme would change the pixel value by +- 10.

Below we show an example how the transformation would be like. During running time,
shifting, rotatiting, shearing and PCA color shifting will all be applied to the image.

![Examples of transformed data.][image4]