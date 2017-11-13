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
[image_pca]: ./figures/alexnet_rgb_pca.png "AlexNet Fancy PCA"
[image4]: ./figures/fig4_rgb_noise.png "Distribution of RGB Noise"
[image5]: ./figures/fig5_data_aug.png "Examples of Transformed Images"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Here is a link to my [project Ipython notebook](https://github.com/juiyangchang/CarND-Term1-Traffic-Sign-Classifer/blob/master/Traffic_Sign_Classifier_v1.ipynb).
You can also check [here]() to see the exported .html file of my notebook.

**Data Set Summary & Exploration**

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I use `numpy.ndarray.shape` and `len(set(y_train))` (for counting the number of unique classes) to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

In the following we show 2 examples for each of the 43 sign classes in 86 panels. As can be seen from the figure below, the images of the same class can contain signs of differeing size, shape, bightness, contrast and even sharpness. 

![Examples of Traffic Signs][image1]

#### 3. Histograms of the Labels.

Here we plot the histogram for all three sets of data.
From the top to the bottom panel, we show
histograms of class labels in the training set, the validation set and the test set, respectively.
It sesms that the training set's distribution more closely resembles that of the test set. For instance, the validation set has equal number of cases in clases 20 to 23 while the training and test sets share similar amount in those classes. Overall, the distributions of the three sets of data are fairly similar.

![Histograms of the Labels. Top panel: histogram of class labels in the training set. Middle panel: histogram of class labels in the validation set. Bottom panel: histogram of class labels in the test set.][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

##### Preprocessing

During data preprocessing, we process the image by first converting the image from RGB to YUV, the equations I followed can be found on the [wiki](https://en.wikipedia.org/wiki/YUV#Conversion_to.2Ffrom_RGB):

* Y = 0.299 R + 0.587 G + 0.114 B 
* U = 0.492 B - Y 
* V = 0.877 R - Y 

(Note that I made a mistake here that the U channel actually approximately 0.492(B - Y), or, -0.147R -0.288G + 0.436 B.  Similar mistake was made with the V channel.  But I have been using this pipeline and can only leave the correct preprocessing into future work.) The Y channel, which is the brightness channel, is then histogram equalized to enhance the contrast with `cv2.equalizeHist`.  In [Sermanet and LeCun (2011)](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), they considered similar preprocessing pipeline where they would first convert image to YUV and then "The  Y  channel is  then preprocessed with  global and local contrast normalization" (quote from their paper), while U and V channels were left unchanged. Finaly we normalize each channel by `X = (X - 128) / 128.`, this will make each channel be within the range of [-1, 1].
The final data dimension for each image is `(32, 32, 3)`.

Below I show an example of the preprocessed image.
![Example of preprocessed image.][image3]

Another preprocessing step we do is to one-hot encode the y labels.

##### Data Augmentation

While our training set data is of moderate size, chances are that we can still be short of data when we try going deep. As can be seen from previous plots, real world images can be taken from different standing points, viewing angles and during a different day time or weather. In the following, I desribe how I use the `ImageDataGenerator` API provided by [keras](https://keras.io/preprocessing/image/) to perform random linear transformations (shifting, rotating, and shearing) to the training images.  In addtion, I also follow the PCA proecdure described in the [AlexNet paper](https://www.cs.toronto.edu/~kriz/imagenet_classification_with_deep_convolutional.pdf) to purturb image intensities in RGB channel.

##### *Linear Transformation with keras.preprocessing.image.ImageDataGenerator*

The `ImageDataGenerator` is an API defined in `keras.preprocessing.image.py` and can be imported with the following line:
```python
from keras.preprocessing.image import ImageDataGenerator
```
To begin with I create an `ImageDataGenerator` object `datagen` with the following transformation types and ranges: rotate by [-15, 15] degrees, vertically and horizontally shift by [-2, 2] pixels, and shear by [-10, 10] degrees.

The first two types of transformations were used in the traffic sign recognition paper ([Sermanet and LeCun, 2011](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)).  They also did image scaling in the paper but
```ImageDataGenerator``` would scale (by setting ```zoom_range``` to a number or a list) along height and width with two different scaling ratios, thus unnaturally deform the sign shape.  Image scaling can be done manually with a transformation matrix or with opencv3 library but I didn't consider going in that direction.  The transforamtion
ranges are hyperparameters potentially worth tuning.

Sometimes datagen would be fitted to training set (`datagen.fit(X_train)`) prior to applying the trasnformations but since our transformations don't depend on the population statistics of the training data we don't need to do it here. 

During trainging we can run something like the following to train over training data with
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
``datagen.flow()``` is a generator that will run indefinitely, which is why we have to stop the loop by checking number of batches trained over. In addtion to the transformations, by default it will also random shuffle the data.

##### ***Color-channel Purturbation with PCA***

Another transformation we consider is to purturb the RGB channels by adding noises. We follow the procedure mentioned in the [AlexNet paper](https://www.cs.toronto.edu/~kriz/imagenet_classification_with_deep_convolutional.pdf).

To begin with we compute the covariance matrix of the RGB channels. Next we compute the principal compontents `lam` and their feature directions `P` with eigendecompostion.  Following the AlexNet paper, during training we add noise to each of RGB channels according to

![][image_pca]

where p_i (or, `P[:,[i]]`) is the eigenvector of the covariance matrix and lambda_i (or, `lam[i]`) is the square root of the eigenvalues. alpha_i's are gaussain random variables drawn independelty for each image. Unlike AlexNet, we use
the lambda_i as square root of the eigenvalues as this would ensure the added noise would have the same covariance matrix as the image when alpha_i's are standard normal.

In AlexNet paper, alpha_i's are of standard deviation 0.1, here I use standard deviation of 0.05, which, as shown in the following plot, in the extreme would change the pixel value by +- 10 (recall that the pixel values range from 0 to 255.)

![][image4]

##### *Putting transformations altogether to perform data augmentation*

Below we show an example how the transformation would be like. During running time,
shifting, rotatiting, shearing and RGB color shifting will all be applied to the image. In panel (a), we show an image in the traning set, in panels (b)-(k), ten
examples of the transformed images are shown.  As can be seen the figure, the image shape can be tilted, rotated, and slightly moved.  The purturbed RGB noise
is not significant, though.

![Examples of transformed data.][image5]

During traning, data augmentation procedure is first
adopted to generate transformed images, then these images are further preprocessed (to YUV coding with Y
histogram equalized).  Then the data is feed into
traing prcedure.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I tried three different network structures: GooLeNet-like, VGG16-like, LeNet5-like.  None of these are the actual network strcture proposed. 

***GoogLeNet-Like***

The GoogLeNet-Like looks like this the table below.
It has some similarity with the structure in the [CVPR 2015 GoogLeNet paper](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf), but is by no means as powerful as the GoogLeNet.
I didn't train a deeper network as I did the training on my laptop.

The convolutional layer includes convolution, batch normalization and relu activation.  Batch normalization
is known to make the joint distribution of preactivation
of different neurons less eclipical, thus helping the progress of gradient descent. Batch normalization is also applied to the fully connected layer. If not specified, I always use `'SAME'` padding in convolutional and pooling layers.

The inception module has four paths: 1x1, 3x3, 5x5 and max pooling.  In the table, # 1x1 indicates the depth of the 1x1 convolution path.  # 3x3 reduce and # 5x5 reduce are the depth of the 1x1 convolution layers applied prior to the 3x3 and 5x5 convolutionaly layers.
The max pooling path applies 3x3 max pooling with stride 1, followed by a 1x1 convolutional layer (which is called pool proj) in the table.  The output depth
of the inception module is # 1x1 + # 3x3 + # 5x5 + # pool proj

| Layer    | patch size/stride | Output | # 1x1|  # 3x3 reduce | # 3x3 | # 5x5 reduce | # 5x5 | # pool proj| 
|:-------------:|:-------------:|:-----:|:----:|:----:|:----:|:----:|:----:|:----:|
| Input         | | 32x32x3|   |  |  |   |   |  |
| convolution   | 3x3/1 | 32 x 32 x 64|   |  |  |   |   |  |
| convolution   | 3x3/1 | 32 x 32 x 64|   |  |  |   |   |  |
| max pooling    | 3x3/2 | 16x16x64|   |  |  |   |   |  |
| inception module |      | 16x16x128| 32 | 32| 64 | 8  | 16   | 16  |
| inception module |      | 16x16x256| 64 | 48| 128 | 16  | 32   | 32  |
| max pooling    | 3x3/2 | 8x8x256|   |  |  |   |   |  |
| inception module |      | 8x8x512| 128 | 64| 256 | 24  | 64   | 64  |
| inception module |      | 8x8x512| 128 | 64| 256 | 24  | 64   | 64  |
| avg pooling*  | 8x8/1  |  1x1x512|   |  |  |   |   |  |
| flatten  |      |   512   |   |  |  |   |   |    |
| dropout (50 %) |   | 512  |   |  |  |   |   |    |
| fully connected linear |        | 43  |   |  |  |   |   |    |
| softmax  |   |43   |  |  |   |   |    |
\* Valid Padding

***VGG-16-Like***


|  Layer  |  patch size/stride | Output |
|:-------:|:-------:|:-------:|
| Input     |     | 32x32x3   |
| convolution| 3x3/1 | 32x32x64|
| convolution| 3x3/1 | 32x32x64|
| max pooling| 2x2/2 | 16x16x64|
| convolution| 3x3/1 | 16x16x128|
| convolution| 3x3/1 | 16x16x128|
| max pooling| 2x2/2 | 8x8x128|
| convolution| 3x3/1 | 8x8x256|
| convolution| 3x3/1 | 8x8x256|
| convolution| 3x3/1 | 8x8x256|
| max pooling| 2x2/2 | 2x2x256|
| flatten |     |  1024|
| fully connected relu | | 512|
|dropout (50%) |  | 512|
| fully connected relu | | 512|
|dropout (50%) |  | 512|
| fully connected linear| | 43|
| softmax  |   |43   |

