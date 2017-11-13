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
[image6]: ./figures/fig6_acc_loss_plot.png "Plot of Loss and Accuracy during Training"
[image7]: ./figures/fig7_test_images.png "Images from the Web"
[image8]: ./figures/fig8_vgg_precision_recall.png "Precision and Recall Bar Chart of VGG-16-Like"
[image9]: ./figures/fig9_googlenet_precision_recall.png "Precision and Recall Bar Chart of GoogLeNet-16-Like"
[image10]: ./figures/fig10_vgg16_feature_map.png "VGG-16-Like's Feature Maps"
[image11]: ./figures/fig11_googlenet_feature_map.png "GoogLeNet-Like's Feature Maps"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Here is a link to my [project Ipython notebook](https://github.com/juiyangchang/CarND-Term1-Traffic-Sign-Classifer/blob/master/Traffic_Sign_Classifier_v1.ipynb).
You can also check [here](https://github.com/juiyangchang/CarND-Term1-Traffic-Sign-Classifer/blob/master/Traffic_Sign_Classifier_v1.html) to see the exported .html file of my notebook.

**Data Set Summary & Exploration**

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I use `numpy.ndarray.shape` and `len(set(y_train))` (for counting the number of unique classes) to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

In the following we show 2 examples for each of the 43 sign classes in 86 panels. As can be seen from the figure below, the images of the same class can contain signs of differing size, shape, brightness, contrast and even sharpness. 

![Examples of Traffic Signs][image1]

#### 3. Histograms of the Labels.

Here we plot the histogram for all three sets of data. From the top to the bottom panel, we show histograms of class labels in the training set, the validation set and the test set, respectively. It seems that the training set's distribution more closely resembles that of the test set. For instance, the validation set has equal number of cases in classes 20 to 23 while the training and test sets share similar amount in those classes. Overall, the distributions of the three sets of data are fairly similar.

![Histograms of the Labels. Top panel: histogram of class labels in the training set. Middle panel: histogram of class labels in the validation set. Bottom panel: histogram of class labels in the test set.][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

##### Preprocessing

During data preprocessing, we process the image by first converting the image from RGB to YUV, the equations I followed can be found on the [wiki](https://en.wikipedia.org/wiki/YUV#Conversion_to.2Ffrom_RGB):

* Y = 0.299 R + 0.587 G + 0.114 B 
* U = 0.492 B - Y 
* V = 0.877 R - Y 

(Note that I made a mistake here that the U channel is actually approximately 0.492(B - Y), or, -0.147R -0.288G + 0.436 B.  Similar mistake was made with the V channel.  But I have been using this pipeline and can only leave the correct preprocessing into future work.) The Y channel, which is the brightness channel, is then histogram equalized to enhance the contrast with `cv2.equalizeHist`.  
This three channels are concatenated into `X`.
In [Sermanet and LeCun (2011)](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), they considered similar preprocessing pipeline where they would first convert image to YUV and then "The  Y  channel is  then preprocessed with  global and local contrast normalization" (quote from their paper), while U and V channels were left unchanged. Finally we normalize each channel by `X = (X - 128) / 128.`, this will make each channel be within the range of [-1, 1].
The final data dimension for each image is `(32, 32, 3)`.

Below I show an example of the preprocessed image.
![Example of preprocessed image.][image3]

Another preprocessing step we do is to one-hot encode the y labels.

##### Data Augmentation

While our training set data is of moderate size, chances are that we can still be short of data when we try going deep. As can be seen from the plot of examples of the training cases, real world images can be taken from different standing points, viewing angles and during a different day time or weather. In the following, I describe how I use the `ImageDataGenerator` API provided by [keras](https://keras.io/preprocessing/image/) to perform random linear transformations (shifting, rotating, and shearing) to the training images.  In addition, I also follow the PCA procedure described in the [AlexNet paper](https://www.cs.toronto.edu/~kriz/imagenet_classification_with_deep_convolutional.pdf) to perturb image intensities in RGB channel.

##### *Linear Transformation with keras.preprocessing.image.ImageDataGenerator*

The `ImageDataGenerator` is an API defined in `keras.preprocessing.image.py` and can be imported with the following line:
```python
from keras.preprocessing.image import ImageDataGenerator
```
To begin with I create an `ImageDataGenerator` object `datagen` with the following transformation types and ranges: rotate by [-15, 15] degrees, vertically and horizontally shift by [-2, 2] pixels, and shear by [-10, 10] degrees.

The first two types of transformations were used in the traffic sign recognition paper ([Sermanet and LeCun, 2011](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)).  They also did image scaling in the paper but
```ImageDataGenerator``` would scale (by setting ```zoom_range``` to a number or a list) along height and width with two different scaling ratios, thus unnaturally deform the sign shape.  Image scaling can be done manually with a transformation matrix or with opencv3 library but I didn't consider going in that direction.  The transformation
ranges are hyperparameters potentially worth tuning.

Sometimes `datagen` would be fitted to training set (`datagen.fit(X_train)`) prior to applying the transformations but since our transformations don't depend on the population statistics of the training data we don't need to do it here. 

During training we can run something like the following to train over training data with
an augmentation factor of 10 (10 times the training set size per epoch)
```python
counter, batches = 0, math.ceil(10 * X_train.shape[0] / 64)   # augment the data by a factor of 10
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=64):
    # do the training
    ...
    counter += 1
    if counter == batches:
        break
    
```
``datagen.flow()``` is a generator that will run indefinitely, which is why we have to stop the loop by checking number of batches trained over. In addition to the transformations, by default it will also random shuffle the data.

##### ***Color-channel Perturbation with PCA***

Another transformation we consider is to purturb the RGB channels by adding noises. We follow the procedure mentioned in the [AlexNet paper](https://www.cs.toronto.edu/~kriz/imagenet_classification_with_deep_convolutional.pdf).

To begin with we compute the covariance matrix of the RGB channels. Next we compute the principal components `lam` and their feature directions `P` with eigendecomposition.  Following the AlexNet paper, during training we add noise to each of RGB channels according to

![][image_pca]

where p_i (or, `P[:,[i]]`) is the eigenvector of the covariance matrix and lambda_i (or, `lam[i]`) is the square root of the eigenvalues. alpha_i's are gaussain random variables drawn independtly for each image. Unlike the AlexNet paper, we use
the lambda_i as square root of the eigenvalues as this would ensure the added noise would have the same covariance matrix as the image when alpha_i's are standard normal.

In the AlexNet paper, alpha_i's are of standard deviation 0.1, here I use standard deviation of 0.05, which, as shown in the following plot, in the extreme would change the pixel value by +- 10 (recall that the pixel values range from 0 to 255.)

![][image4]

##### *Putting transformations altogether to perform data augmentation*

Below we show an example of how the transformation would be like. During running time,
shifting, rotating, shearing and RGB color shifting will all be applied to the image. In panel (a), we show an image in the training set, in panels (b)-(k), ten
examples of the transformed images are shown.  As can be seen in the figure, the image shape can be tilted, rotated, and slightly moved.  The perturbed RGB noise
is not significant, though.

![Examples of transformed data.][image5]

During training, data augmentation procedure is first
adopted to generate transformed images, then these images are further preprocessed (to YUV coding with Y
histogram equalized).  Then the data is feed into
training procedure.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I tried three different network structures: GooLeNet-like, VGG016-like, LeNet5-like.  None of these are the actual network structure proposed in the literature, instead they draw ideas from the literature. I would say both GooLeNet-like and VGG-16-Like networks are both my final models.

In the following I describe each of the three network structures.

***GoogLeNet-Like***

The GoogLeNet-Like network looks like the table listed below.
It has some similarity with the structure in the [CVPR2015 GoogLeNet paper](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf), but is by no means as powerful as the GoogLeNet.
I didn't train a deeper network as I did the training on my laptop.

The convolution layer includes convolution, batch normalization and relu activation.  Batch normalization
is known to make the joint distribution of pre-activation
of different neurons less elliptical, thus helping the progress of gradient descent.  If not specified, I always use `'SAME'` padding in convolutional and pooling layers.

Batch normalization is also applied to the fully connected layer. A typiccal fully connected layer includes
matrix multiplication, batch normalization and neuronal activation. I would specify the type of activation function used in the table entry for the fully connected layer. Here the only fully connected layer has linear activation, its activation is further fed into softmax function for making prediction of class probabilities.

The inception module has four paths: 1x1, 3x3, 5x5 and max pooling.  In the table, # 1x1 indicates the depth of the 1x1 convolution path.  # 3x3 reduce and # 5x5 reduce are the depths of the 1x1 convolution layers applied prior to the 3x3 and 5x5 convolution layers.
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

Another network structure I tried is the VGG-16-like network.  As can be seen from the table below, it's structure kind of resembles that of [VGG team's 16 layer network in 2014](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md). But it only contains 10 weighted layers.


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

***LeNet5-Like***

The last network I tried is LeNet5-Like. It differs
from the original LeNet5 in that it uses `'SAME'` padding instead of `'VALID'` padding. In addition it
also uses batch normalization and dropout.


|  Layer  |  patch size/stride | Output |
|:-------:|:-------:|:-------:|
| Input     |     | 32x32x3   |
| convolution| 5x5/1 | 32x32x6|
| max pooling| 2x2/2 | 16x16x6|
| convolution| 5x5/1 | 16x16x16|
| max pooling| 2x2/2 | 8x8x16|
| flatten |     | 1024|
| fully connected relu | | 120|
|dropout (50%) |  |120|
| fully connected relu | | 84|
|dropout (50%) |  |84|
| fully connected linear| | 43|
| softmax  |   |43   |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I trained the VGG-16-like and LeNet5-like networks for 50 epochs of non-augmented data. Each epoch is a full run over all data points in the training set. I always use batch size of 64. I tried briefly with batch sizes of 32 but the loss fluctuates a lot.  In reflection, 32 can be too small for the batch size as we won't see all classes in a batch. I never tried larger batch sizes.

For the GoogLeNet-Like structure, I trained over 40 epochs of augmented data.  I augmented the training data
by a factor of three, meaning that in an epoch, for each training image, the original image and two transformed images of the image are used in training.
So in each epoch, we training over 104,397 images.

I used ADAM Optimizer for all three networks with learning rate of 0.003.  ADAM optimizer is known to 
be more robust as it will automatically tune the learning rate with momentum.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The figure below depicts the training and validation set performance during training.  The plots of the GoogLeNet-like network is stretched in x axis as
it effectively sees three times the data points in an epoch.  It is quite apparently the LeNet5-like is suffering overfitting. The rest two is kind of similar performance-wise.


![][image6]

Below we report the final loss and accuracy over the training and validation sets.  Apparently the LeNet5-like network suffer from larger scale of overfitting.  It is not clear if the rest two network structures suffer from overfitting but it seems also applies for the two as well.  But VGG-16-Like seems to be a better network in terms of the validation accuracy.


|      | Training Loss        | Validation Loss |         Training Accuracy         |  Validation Accuracy       |
| ------------- |:-------------:| --------:|--------:|--------------:|
| VGG-16-Like     |0.1376  | 0.1510 |   1    |   0.9966                  |
| GoogLeNet-Like      | 0.1256      |   0.1457 |   1  | 0.9955      |
| LeNet5-Like | 0.2903      |     0.5597 |  0.9992     |    0.9599   |

We also evaluate test set loss and accuracy for the three models.  VGG-16-like is apparently the best. Comparatively, the GoogLeNet-like network made roughly 18 more mistakes.

|      | Loss        | Accuracy  |
| ------------- |:-------------:| -----:|
| VGG-16-Like     |0.1915  | 0.9908 |
| GoogLeNet-Like      | 0.1852      |   0.9894 |
| LeNet5-Like | 0.5995      |    0.9468 |

My final model results were:
* training set accuracy of  1
* validation set accuracy of 0.9966
* test set accuracy of 0.9908
 
 My approach for choosing network structure was kind of iterative and
 partially using well known architectures. I initially use LeNet5-like network.
 With batch normalization and dropout, LeNet5-like network already achieves
 the project goal. 
 
 Batch normalization is adopted as it is known to accelerate the training and increase accuracy.  Batch normalization is used in preactivation of convolution and fully connected layers.  Dropout is applied to as a form of regularization. Dropout is only added after the activation of a fully connected layer (except for the classification layer).  It seems to be a common practice to not use dropout after convolution layer.  For one thing,
 the convolution layer is already sparse as each neuron only connect to a small
 patch from the previous layer.  Each neuron here is a particular pixel in a layer of feature.  I did try using dropout after convolution layer briefly but
 it seems that the training accuracy would fluctuate a lot and the improvement is insignificant.  Dropout rate is set to 0.5 and I did not consider tuning it.

My other design choice was to use `SAME` padding instead of `VALID` padding in LeNet5-like network. I don't know if it has encouraged a better performance, but it seems that `SAME` padding is used in modern networks more. Moreover, I found it easier for keep tracking of the output sizes.

 I moved on to adding more convolution layers before each pooling layer, as it seems to be more common to use multiple convolution layers in modern designs, according to the [CS231n course notes](http://cs231n.stanford.edu/).  During the trials, I also noticed that adding more layers to the initial convolution layers is helpful (and also increase the depth of later layers as later layers typically increase depths from the former layers).  I also noticed that the changing kernel size from 5 to 3 seems won't degrade the performance.  Furthermore, kernel size of 5 is considerably slower than 3 in deeper structures.  I ended up just using a part of [the VGG team's 16-layer network](https://arxiv.org/pdf/1409.1556).
 Interestingly, the VGG-16-like network suppresses the overfitting issue from
 the LeNet5-like structure greatly.
 I also tried replacing part of the convolution layers in the VGG-16-like networks to inception modules, which I called the GoogLeNet-like.  The dimension of the layers take ideas from the [GoogLeNet paper](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf).  The main difference from them is that I used batch normalization in each of the weighted layers (convolution and fully connected).

 As a parting word, convolution neural network should be suitable to traffic sign problem as the signs can be appearing in different locations in the images, and taken with differing viewing points and distances. Convolution
 layers are shift invariant so it should be able to learn useful common features
 occurring at differing locations in two or more images.  The reasoning behind using VGG 16-layer or the inception modules is more of a random decision.  But 
 I think it always worth trying competition winning network structures. My final networks -- the VGG-16-like and the GoogLeNet-like networks are both quite good performance-wise.  It would be interesting to see if the error rate
 over the validation sets can be further decreased.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:
 ![][image7]

 The first image might be difficult to classify because it is being viewed from a bottom angle and slightly smaller than the whole image.  The second image is
 also kind of difficult as it shares the same problem of being smaller than the image. Finally the fifth image can potentially be difficult as it is kind of blurred by surrounding noises (snow, it is, as the name of the sign suggests).

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction (VGG-16-Like)        					|  Prediction (GoogLeNet-Like) |
|:---------------------:|:---------------------------------------------:|:---:|
| Vehicles over 3.5 metric tons prohibited      		| Yield   									| Vehicles over 3.5 metric tons prohibited      		| Yield   							|
| Bumpy road     			| Speed limit (20km/h)										| Bumpy road     			|
| Priority road					| Priority road											| Priority road											|
| Children crossing	      		| Children crossing					 				| Children crossing					 				|
| Beware of ice/snow			| Beware of ice/snow  | Speed limit (100km/h) |

Here we show the accuracy rates for the two networks over the test images from the web:

|      |  Accuracy  |
|:-----|:-----------|
| VGG-16-Like | 0.6 |
| GoogLeNet-Like | 0.8|


The *VGG-16-like* model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%.  This is comparatively worse than the test set performance.  

It is kind of difficult to reason why it would decide the first image as yield. For one thing, yield is a triangular sign while the sign in the image is circular. Maybe it was because the bottom rectangular shape in the image.  The precision and recall rates of classes 16 (Vehicles over 3.5 metric tons prohibited ) and 13 (Yield) aren't particularly low or high as in the bar chart below.  The VGG-16-like model also failed to classify the second image correctly, this time mistaken a sign of triangular shape with a sign of circular shape (Speed Limit).  Again the precision and recall charts aren't useful in reasoning this mistake.  For both cases I would think it is because
these two images doesn't resemble those the network saw during training (differing viewing angle and size).



![][image8]

The *GoogLeNet-like* structure on the other hand classifies 80% of the images correctly. This is still lower than the test set accuracy but after all we only have five images here.  It made mistake on class 30 (Beware of ice/snow).  It is not clear from the bar chart below if the precision/recall rate of the classes contributed to the error, though.  Consider adding local jittering noises instead of global RGB noise in image augmentation during training might be helpful.

![][image9]

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Below I describe the prediction probabilities of the five new images for both networks.

***VGG-16-Like***

In test case 1,  the actual label is class 16 (Vehicles over 3.5 metric tons prohibited).  The top five probabilities can be seen below.  The actual class 
is not even in top five but the network seems to be not really certain about
the label for this image anyways as the prediction probability are all kind of small.  Note that the second to fifth labels are all of circular shape, same 
as the actual label.

|Probability|Prediction|
|:----:|:----:|
|0.179|13 (Yield)|
|0.083|5 (Speed limit (80km/h))|
|0.052|38 (Keep right)|
|0.048|2 (Speed limit (50km/h))|
|0.042|9 (No passing)|

For test case 2, the actual label is class 22 (Bumpy road).  The top five probabilities are listed in the table below.  The actual class is not in top five again and this time  it is a hard mistake as it is quite certain the image
is label 0.  Three out of five of the images are circular shape, maybe it was due to the bottom snow-like pile in the image being of circular shape.

|Probability|Prediction|
|:----:|:----:|
|0.753|0 (Speed limit (20km/h))|
|0.015|12 (Priority road)|
|0.014|1 (Speed limit (30km/h))|
|0.013|38 (Keep right)|
|0.013|13 (Yield)|

Test Case 3, Actual Label: 12 (Priority road): this one is less interesting as the network is correct.

|Probability|Prediction|
|:----:|:----:|
|0.771|12 (Priority road)|
|0.016|2 (Speed limit (50km/h))|
|0.015|1 (Speed limit (30km/h))|
|0.014|38 (Keep right)|
|0.014|13 (Yield)|

Test Case 4, Actual Label: 28 (Children crossing): this one is less interesting as the network is correct.

|Probability|Prediction|
|:----:|:----:|
|0.978|28 (Children crossing)|
|0.001|2 (Speed limit (50km/h))|
|0.001|1 (Speed limit (30km/h))|
|0.001|38 (Keep right)|
|0.001|12 (Priority road)|

Test Case 5, Actual Label: 30 (Beware of ice/snow): Despite making the right
judgment, the network is comparatively less certain about this image. Maybe it is due to the fact that of class 30 having a lower recall rate for VGG-16-like network.

|Probability|Prediction|
|:----:|:----:|
|0.249|30 (Beware of ice/snow)|
|0.078|7 (Speed limit (100km/h))|
|0.048|11 (Right-of-way at the next intersection)|
|0.041|1 (Speed limit (30km/h))|
|0.040|2 (Speed limit (50km/h))|

***GoogLeNet-Like***

Test Case 1, Actual Label: 16 (Vehicles over 3.5 metric tons prohibited): Unlike the VGG-16-like network, the GoogLeNet-like is quite certain about this one -- and it is right.  

|Probability|Prediction|
|:----:|:----:|
|0.910|16 (Vehicles over 3.5 metric tons prohibited)|
|0.006|2 (Speed limit (50km/h))|
|0.006|38 (Keep right)|
|0.006|13 (Yield)|
|0.005|1 (Speed limit (30km/h))|

Test Case 2, Actual Label: 22 (Bumpy road): Unlike the VGG-16-like network, the GoogLeNet-like is quite certain about this one -- and it is right. It is probably worth investigating if it was due to image augmentation or inception module.  Recall that the VGG-16-like network wasn't trained with image augmentation. But I would guess it is more due to data augmentation as I did apply shifting and shearing in data augmentation process.

|Probability|Prediction|
|:----:|:----:|
|0.999|22 (Bumpy road)|
|0.000|2 (Speed limit (50km/h))|
|0.000|13 (Yield)|
|0.000|10 (No passing for vehicles over 3.5 metric tons)|
|0.000|4 (Speed limit (70km/h))|

Test Case 3, Actual Label: 12 (Priority road): this one is less interesting as the network is correct.

|Probability|Prediction|
|:----:|:----:|
|0.739|12 (Priority road)|
|0.018|2 (Speed limit (50km/h))|
|0.017|13 (Yield)|
|0.017|1 (Speed limit (30km/h))|
|0.016|10 (No passing for vehicles over 3.5 metric tons)|

Test Case 4, Actual Label: 28 (Children crossing): this one is less interesting as the network is correct.

|Probability|Prediction|
|:----:|:----:|
|0.960|28 (Children crossing)|
|0.003|2 (Speed limit (50km/h))|
|0.003|1 (Speed limit (30km/h))|
|0.003|13 (Yield)|
|0.002|10 (No passing for vehicles over 3.5 metric tons)|

Test Case 5, Actual Label: 30 (Beware of ice/snow): The network kind of has no idea what this is.  I don't know if it was because of the jittering noise in the image being unfamiliar to the network.

|Probability|Prediction|
|:----:|:----:|
|0.097|7 (Speed limit (100km/h))|
|0.082|5 (Speed limit (80km/h))|
|0.073|2 (Speed limit (50km/h))|
|0.060|12 (Priority road)|
|0.051|1 (Speed limit (30km/h))|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Below I show the feature map of the first pooling layers in the VGG-16-like and
GoogLeNet-like networks when applied with the first test image from the web.
Both networks conclude that circular shapes is probably important for classifying this image.  One thing worth noticing is that, the GoogLeNet feature maps, despite being more blurred, is less likely to react to a local area in the background. On the other hand, there are multiple feature maps
in the VGG-16-like network being reactive to the top left corner.  I think this
provides a proof of the image augmentation being helpful in fitting to local noises. The networks also seem to be trying to identify the truck in the middle
of the sign as we can see multiple feature maps showing a white block in the middle of surrounding circle.

***VGG-16-like***
![][image10]

***GoogLeNet-like***
![][image11]