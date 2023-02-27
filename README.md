![Banner](https://github.com/cla-cif/Cherry-Powdery-Mildew-Detector/blob/main/readme_images/banner.jpg)

## Table of Contents
1. [Dataset Content](#dataset-content)
2. [Business Requirements](#business-requirements)
3. [Hypothesis and validation](#hypothesis-and-validation)
4. [Rationale for the model](#the-rationale-for-the-model)
5. [Trial and error](#trial-and-error)
6. [Implementation of the Business Requirements](#the-rationale-to-map-the-business-requirements-to-the-data-visualizations-and-ml-tasks)
7. [ML Business case](#ml-business-case)
8. [Dashboard design](#dashboard-design-streamlit-app-user-interface)
9. [CRISP DM Process](#the-process-of-cross-industry-standard-process-for-data-mining)
10. [Bugs](#bugs)
11. [Deployment](#deployment)
12. [Technologies used](#technologies-used)
13. [Credits](#credits)

### Deployed version at [cherry-powdery-mildew-detector.herokuapp.com](https://cherry-powdery-mildew-detector.herokuapp.com/)

## Dataset Content

The dataset contains 4208 featured photos of single cherry leaves against a neutral background. The images are taken from the client's crop fields and show leaves that are either healthy or infested by powdery mildew a biotrophic fungus. This disease affects many plant species but the client is particularly concerned about their cherry plantation crop since bitter cherries are their flagship product. 

## Business Requirements

We were requested by our client Farmy & Foods a company in the agricultural sector to develop a Machine Learning based system to detect instantly whether a certain cherry tree presents powdery mildew thus needs to be treated with a fungicide. 
The requested system should be capable of detecting instantly, using a tree leaf image, whether it is healthy or needs attention. 
The system was requested by the Farmy & Food company to automate the detection process conducted manually thus far. The company has thousands of cherry trees, located on multiple farms across the country. As a result, this manual process is not scalable due to the time spent in the manual process inspection.
Link to the wiki section of this repo for the full [business interview](https://github.com/cla-cif/Cherry-Powdery-Mildew-Detector/wiki/Business-understanding-interview). 

Summarizing:

1. The client is interested in conducting a study to visually differentiate a healthy cherry leaf from one infected by powdery mildew.
2. The client is interested in predicting if a cherry tree is healthy or contains powdery mildew.
3. The client is interested in obtaining a prediction report of the examined leaves. 

## Hypothesis and validation

1. **Hypothesis**: Infected leaves have clear marks differentiating them from the healthy leaves.
   - __How to validate__: Research about the disease and build an average image study can help to investigate it.<br/>

2. **Hypothesis**: Mathematical formulas comparison: `softmax` performs better than `sigmoid` as activation function for the CNN output layer. 
   - __How to validate__: Understand the kind of problem we are trying to solve and the differences between matemathical functions used to solve that class of problem. Train and compare identical models changing only the activation function of the output layer. <br/>

3. **Hypothesis**: Converting `RGB` images to `grayscale` improves image classification performance.  
   - __How to validate__: Understand how colours are represented in tensors. Train and compare identical models changing only the image color.

### Hypothesis 1
> Infected leaves have clear marks differentiating them from the healthy leaves.

**1. Introduction**

We suspect cherry leaves affected by powdery mildew have clear marks, typically the first symptom is a light-green, circular lesion on either leaf surface, then a subtle white cotton-like growth develops in the infected area. 

**2. Observation**

An Image Montage shows the evident difference between a healthy leaf and an infected one. 

![montage_healthy](https://github.com/cla-cif/Cherry-Powdery-Mildew-Detector/blob/main/readme_images/montage_healthy.png)
![montage_infected](https://github.com/cla-cif/Cherry-Powdery-Mildew-Detector/blob/main/readme_images/montage_infected.png)

Difference between average and variability images shows that affected leaves present more white stipes on the center.

![average variability between samples](https://github.com/cla-cif/Cherry-Powdery-Mildew-Detector/blob/main/readme_images/average_image.png)

While image difference between average infected and average infected leaves shows no intuitive difference. 

![average variability between samples](https://github.com/cla-cif/Cherry-Powdery-Mildew-Detector/blob/main/readme_images/avg_diff.png)

**3. Conclusion**

The model was able to detect such differences and learn how to differentiate and generalize in order to make accurate predictions.
A good model trains its ability to predict classes on a batch of data without adhering too closely to that set of data.
In this way the model is able to generalize and predict future observation reliably because it didn't 'memorize' the relationships between features and labels as seen in the training dataset but the general pattern from feature to labels.

**Sources**:

- [Pacific Northwest Pest Management Handbooks](https://pnwhandbooks.org/plantdisease/host-disease/cherry-prunus-spp-powdery-mildew)

---
### Hypothesis 2
> Mathematical formulas comparison: `softmax` performs better than `sigmoid` as activation function for the CNN output layer. 
For further details the results mentioned in this section can be downloaded here [softmax hypothesis](https://github.com/cla-cif/Cherry-Powdery-Mildew-Detector/blob/main/attachments/ModellingEvaluating_softmax_rgb.ipynb) and here [sigmoid hypothesis](https://github.com/cla-cif/Cherry-Powdery-Mildew-Detector/blob/main/attachments/ModellingEvaluating_sigmoid.ipynb)

**1. Introduction**

   1. Understand problem and mathematical functions

First of all let's understand the problem our model is asked to solve. The model is required to assign to a cherry leaf one of the two categories: healthy/infected, which makes it a classification problem. It could be seen as a binary classification (healthy vs NOT healthy) or as a multiclass classification where each output is assigned one and only one label from more than two classes (just two in our case: healthy vs infected).

If the problem is seen as **binary classification** we will have 1 output node. The probability of the output belonging to one class or the other is within the range of 0 and 1 so if is probability <0.5 is considered class 0 (healthy) and if probability >=0.5 is considered class 1 (infected).<br/>
These constraints are given by the ```sigmoid``` function which is also called the _squashing_ function as the classes will converge either to 0 or 1. It's computationally effective but used for binary classification problems only as it suffers major drawbacks which include sharp damp gradients during backpropagation. <br/> 
Backpropagation is where the “learning” or “adjustment” takes place in the neural network in order to adjust the weights of all the nodes throughout the layers of the network. The error value (distance between actual and predicted label) flows back through the network in the opposite direction as before, and it is then used in combination with the derivative of the Sigmoid function. <br/>
The derivative of a function will give us the angle/slope of the graph that the function describes. This value will let the network know whether to increase or decrease the value of the individual weights in the layers of the network but for a very high or very low value of the error, the derivative of the sigmoid is very low (hence the _squashing_ effect).  

If we see the problem as **multi class classification** we will have 2 output nodes (because I want to predict two classes healthy vs infected). In this case the `softmax` function is applied to the output layer. Like the previous case the output of this function lies in the range [0,1] but now we are looking at a probability distribution over the predicted classes which adds up to 1 with the target class having the highest probability. The probability distribution comes from normalizing the output for each class between 0 and 1 and divide by their sum. 

   2. Understand how to evaluate the performance
   
A learning curve is a plot of model learning performance over experience or time.
Learning curves are a widely used diagnostic tool in machine learning for algorithms that learn from a training dataset incrementally. The model can be evaluated on the training dataset and on a hold out validation dataset after each update during training and plots of the measured performance can created to show learning curves.
Reviewing learning curves of models during training can be used to diagnose problems with learning, such as an underfit or overfit model, as well as whether the training and validation datasets are suitably representative. <br/>
Generally, a learning curve is a plot that shows time or experience on the x-axis (Epoch) and learning or improvement on the y-axis (Loss/Accuracy).
   -  **Epoch**: refers to the one entire passing of training data through the algorithm. 
   -  **Loss**: Loss is the penalty for a bad prediction. That is, loss is a number indicating how bad the model's prediction was on a single example. If the model's prediction is perfect, the loss is zero; otherwise, the loss is greater. In our case loss on training set was evaluated against loss on validation set.
   -  **Accuracy**: Accuracy is the fraction of predictions our model got right. Again accuracy on the training set was measured against accuracy on the validation set. 

In our plot we will be looking for a *good fit* of the learning algorithm which exists between an overfit and underfit model
A good fit is identified by a training and validation loss that decreases to a point of stability with a minimal gap between the two final loss/accuracy values.
We should expect some gap between the train and validation loss/accuracy learning curves. This gap is referred to as the “generalization gap.”
A plot of learning curves shows a good fit if:
   -  The plot of training loss decreases (or increases if it's an accuracy plot) to a point of stability.
   -  The plot of validation loss decreases/increases to a point of stability and has a small gap with the training loss.
   -  Continued training of a good fit will likely lead to an overfit (That's why ML models usually have a [early stopping](https://en.wikipedia.org/wiki/Early_stopping) which interrupts the model's learning phase when it stops to improve).
  
**2. Observation**

The model was set to train only on 32 Epoch with no early stoppings, just for the purpose of this hypothesis, and shows overfitting around the 10 last epochs as expected.
The same hyperparameters were set for both examples. 
The model trained using ```softmax``` showed less training/validation sets gap and more consistent learning rate after the 5th Epoch compared to the model trained using ```sigmoid```. 
 - Loss/Accuracy of LSTM model trained using `softmax`
 
   ![softmax_model](https://github.com/cla-cif/Cherry-Powdery-Mildew-Detector/blob/main/streamlit_images/model_history_rgb_softmax.png) 
 - Loss/Accuracy of LSTM model trained using `sigmoid`
 
   ![rgb_model](https://github.com/cla-cif/Cherry-Powdery-Mildew-Detector/blob/main/streamlit_images/model_history_sigmoid.png)
   
**3. Conclusion**

In our case the ```softmax``` function performed better. 

**Sources**:
- [Activation Functions Compared With Experiments](https://wandb.ai/shweta/Activation%20Functions/reports/Activation-Functions-Compared-With-Experiments--VmlldzoxMDQwOTQ) by [Sweta Shaw](https://wandb.ai/shweta)
- [Backpropagation in Fully Convolutional Networks](https://towardsdatascience.com/backpropagation-in-fully-convolutional-networks-fcns-1a13b75fb56a#:~:text=Backpropagation%20is%20one%20of%20the,respond%20properly%20to%20future%20urges.) by [Giuseppe Pio Cannata](https://cannydatascience.medium.com/)
- [Understanding The Derivative Of The Sigmoid Function](https://towardsdatascience.com/understanding-the-derivative-of-the-sigmoid-function-cbfd46fb3716#:~:text=The%20Sigmoid%20function%20is%20often,of%20the%20network%20or%20not.) by [Jacob Toftgaard Rasmussen](https://jacobtoftgaardrasmussen.medium.com/)
- [How to use Learning Curves to Diagnose Machine Learning Model Performance](https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/) by [Jason Brownlee](https://machinelearningmastery.com/about)
- [Activation Functions: Comparison of Trends in Practice and Research for Deep Learning](https://arxiv.org/pdf/1811.03378.pdf) by *Chigozie Enyinna Nwankpa, Winifred Ijomah, Anthony Gachagan, and Stephen Marshall*

---
### Hypothesis 3 
> Converting ```RGB``` images to ```grayscale``` improves image classification performance. 

For further details the results mentioned in this section can be downloaded here [rgb hypothesis](https://github.com/cla-cif/Cherry-Powdery-Mildew-Detector/blob/main/attachments/ModellingEvaluating_softmax_rgb.ipynb) and here [gray hypothesis](https://github.com/cla-cif/Cherry-Powdery-Mildew-Detector/blob/main/attachments/ModellingEvaluating_gray.ipynb)

**1. Introduction**

Digital images are made of pixels, every image has three main properties:
   - Size — This is the height and width of an image. It can be represented in centimeters, inches or even in pixels.
   - Color space — Examples are RGB and HSV color spaces.
   - Channel — This is an attribute of the color space. 
  
Each pixel of a coloured image is made of combinations of primary colors represented by a series of code. RGB color space has three types of colors or attributes known as Red, Green and Blue (hence the name RGB).
A grayscale image is one in which the value of each pixel is a single sample representing only an amount of light; that is, it carries only intensity information. Grayscale images, a kind of black-and-white or gray monochrome, are composed exclusively of shades of gray.

In an RGB image where there are three color channels, a pixel value has three numbers, each ranging from 0 to 255 (both inclusive). For example, the number 0 of a pixel in the red channel means that there is no red color in the pixel while the number 255 means that there is 100% red color in the pixel. A single RGB image can be represented using a three-dimensional (3D) NumPy array or a tensor.<br/>
In a grayscale image where there is only one channel, a pixel value has just a single number ranging from 0 to 255 (both inclusive). The pixel value 0 represents black and the pixel value 255 represents white. Therefore a single grayscale image can be represented using a two-dimensional (2D) NumPy array or a tensor because it doesn't need an extra dimension for the color channel. <br/>
Feeding a model with an RGB image or convert that image to grayscale, depends on the nature of the images and the information conveyed by the colour. 
If the color has no significance in the image to classify, indeed a grayscale image requires less computational power to be processed.<br/><br/>

**2. Observation**

The model was set to train only on 32 Epoch with no early stoppings, just for the purpose of this hypothesis, and shows overfitting around the 10 last epochs as expected.
The same hyperparameters were set for both examples. 
The model trained using RGB images showed less training/validation sets gap and more consistent learning rate after the 5th Epoch compared to the model trained using Grayscale images. 
The same CNN applied to an RGB image dataset has 3,715,234 parameters to train compared to 3,714,658 parameters when the same dataset is converted to grayscale. 

   - Comparison of the same infected leaf image 
   
  ![gray_leaf](https://github.com/cla-cif/Cherry-Powdery-Mildew-Detector/blob/main/readme_images/leaf_gray.png) ![rgb_leaf](https://github.com/cla-cif/Cherry-Powdery-Mildew-Detector/blob/main/readme_images/leaf_rgb.png)
  

   - Loss/Accuracy of LSTM model trained on grayscale images
   
   ![gray_model](https://github.com/cla-cif/Cherry-Powdery-Mildew-Detector/blob/main/streamlit_images/model_history_gray.png) 
   - Loss/Accuracy of LSTM model trained on RGB images
   
   ![rgb_model](https://github.com/cla-cif/Cherry-Powdery-Mildew-Detector/blob/main/streamlit_images/model_history_rgb_softmax.png)

**3. Conclusion**

Keeping the colour information performed better. The plot shows lower loss and more consistent accuracy. A difference of 676 trainable parameters has no significant benefit on the computational cost. 

Sources:
- [How RGB and Grayscale Images Are Represented in NumPy Arrays](https://towardsdatascience.com/exploring-the-mnist-digits-dataset-7ff62631766a) by [Rukshan Pramoditha](https://rukshanpramoditha.medium.com/)

## The rationale for the model

The model has 1 input layer, 3 hidden layers (2 ConvLayer, 1 FullyConnected), 1 output layer. 

### The goal

Setting the hyperparameters, determining the number of hidden layers and node, choosing the optimizer was a matter of trial and error. </br> 
The model does not necessarily represent the best one but this structure was eventually chosen evaluating the outcome of multiple tests and tuning the model according to the goal. See [Trial and error](#trial-and-error) </br>

A good model trains its ability to predict classes on a batch of data without adhering too closely to that set of data. In this way the model is able to generalize and predict future observation reliably because it didn't 'memorize' the relationships between features and labels as seen in the training dataset but the general pattern from feature to labels. 

A good model also requires as little as possible computational power by keeping down the neural network complexity and the number of trainable parameters while still being able to generalize, maintain accuracy and minimize error. 

### Choosing the hyperparameters

- **Convolutional layer size**: Using a two dimensions CNN (`Conv2D`) is appropriate for the pictures in our dataset which are not volumetrical (those require a 3D CNN). 1D convolution layer is also not a good fit because creates a convolution kernel that is convolved with the layer input over a single spatial (or temporal) dimension to produce a tensor of outputs. 

- **Convolutional kernel size**: The convolutional filter (3x3) moves across x-axes and y-axes (stride 1 in our case) of the input shape of the images, hence a Conv2D (two dimensions). </br>
The convolutional filter (or kernel) 3x3 works well with a 2D CNN (The third dimension is equal to the number of channels of the input image), better than a 2x2 which won't allow a zero padding because image sizes are even numbers (keeping stride=1) and better than a 5x5 which extracts less information. A small kernel looks at very few pixels at once hence focusing on 'the details'. 

- **Number of neurons**: The chosen numbers are power of 2 due to computational reasons. The GPU can take advantage of optimizations related to efficiencies in working with powers of two.

- **Activation function**: `ReLu` is used is because it is simple, fast, and empirically it seems to work well. It has been observed that training a deep network with `ReLu` tended to converge much more quickly and reliably than training a deep network with `sigmoid` activation. Furthermore, the derivative of ReLu is either 0 or 1, so multiplying by it won't cause weights that are further away from the end result of the loss function to suffer from the vanishing gradient problem. 

- **Pooling**: Pooling is performed in neural networks to reduce variance and computation complexity. Among Average pooling, Min pooling and Max pooling, we chose the latter which selects the brighter pixels from the image. `MaxPooling` is useful when the background of the image is dark (green in our case) and we are interested in only the lighter pixels of the image (powdery mildew is white).

- **Output Activation Function**: This kind of classification problem requires either `softmax` or `sigmoid` activation functions. Our model trained using ```softmax``` showed less training/validation sets gap and more consistent learning rate after the 5th Epoch compared to the model trained using ```sigmoid```. See [Hypothesis 2](#Hypothesis-2) for more details.

- **Dropout**:  The Dropout layer is a mask that nullifies the contribution of some neurons towards the next layer and leaves unmodified all others. Dropout layers are important in training CNNs because they prevent overfitting on the training data. If they aren’t present, the first batch of training samples influences the learning in a disproportionately high manner. Since the number of samples is not extremely high, 20% dropout was deemed appropriate. 

**Source**: 
- [How to choose the size of the convolution filter or Kernel size for CNN?](https://medium.com/analytics-vidhya/how-to-choose-the-size-of-the-convolution-filter-or-kernel-size-for-cnn-86a55a1e2d15) by - [Swarnima Pandey](https://medium.com/@pandeyswarnima)
- [The advantages of ReLu](https://stats.stackexchange.com/questions/126238/what-are-the-advantages-of-relu-over-sigmoid-function-in-deep-neural-networks#:~:text=The%20main%20reason%20why%20ReLu,deep%20network%20with%20sigmoid%20activation.)
- [Maxpooling vs minpooling vs average pooling](https://medium.com/@bdhuma/which-pooling-method-is-better-maxpooling-vs-minpooling-vs-average-pooling-95fb03f45a9#:~:text=Average%20pooling%20method%20smooths%20out,lighter%20pixels%20of%20the%20image.) by - [Madhushree Basavarajaiah](https://medium.com/@bdhuma)
- [How ReLU and Dropout Layers Work in CNNs](https://www.baeldung.com/cs/ml-relu-dropout-layers)

### Hidden Layers

They are “hidden” because the true values of their nodes are unknown in the training dataset as we only know the input and output.</br> 
These layers perform feature extraction and classification based on those features. 

>There are really two decisions that must be made regarding the hidden layers: how many hidden layers to actually have in the neural network and how many neurons will be in each of these layers. 
Using too few neurons in the hidden layers will result in something called underfitting. Underfitting occurs when there are too few neurons in the hidden layers to adequately detect the signals in a complicated data set.
Using too many neurons in the hidden layers can result in several problems. First, too many neurons in the hidden layers may result in overfitting. 

**Source**: *Introduction to Neural Networks for Java* by Jeff Heaton. Preview at [Google Books](https://books.google.it/books?id=Swlcw7M4uD8C&pg=PA158&lpg=PA158&dq=Introduction%20to%20Neural%20Networks%20for%20Java,%20Second%20Edition%20The%20Number%20of%20Hidden%20Layers&source=bl&ots=TJx9QaeWw6&sig=gZqg9e73K1oCqWBxmcBWAf2pbrE&hl=it&sa=X&ved=0CCUQ6AEwAGoVChMIudnOsJr1yAIVwjkaCh3AAgnU#v=onepage&q=Introduction%20to%20Neural%20Networks%20for%20Java%2C%20Second%20Edition%20The%20Number%20of%20Hidden%20Layers&f=false)

>In order to secure the ability of the network to generalize the number of nodes has to be kept as low as possible. If you have a large excess of nodes, you network becomes a memory bank that can recall the training set to perfection, but does not perform well on samples that was not part of the training set. 
[Steffen B Petersen](https://www.researchgate.net/post/How-to-decide-the-number-of-hidden-layers-and-nodes-in-a-hidden-layer)

- **Conv vs FC Layers**: 
  - *Convolutional Layer* are used for feature extraction, use fewer parameters by forcing input values to share the parameters. 
  - *Dense Layers* use a linear operation meaning every output is formed by the function based on every input. They are used as final layers in some models because they can directly perform classification.

**Source**: 
- [Dense Layer vs convolutional layer](https://datascience.stackexchange.com/questions/85582/dense-layer-vs-convolutional-layer-when-to-use-them-and-how#:~:text=As%20known%2C%20the%20main%20difference,function%20based%20on%20every%20input.)


### Model Compilation

- **Loss**: A loss function is a function that compares the target and predicted output values; measures how well the neural network models the training data. When training, we aim to minimize this loss between the predicted and target outputs. `categorical_crossentropy` (also called Softmax Loss. It is a Softmax activation plus a Cross-Entropy loss) was used since the problem has been treated as multiclass classification. See [Hypothesis 2](#Hypothesis-2) for more details.

- **Optimizer**: An optimizer is a function or algorithm that is created and used for neural network attribute modification (i.e., weights, learning rates) for the purpose of speeding up convergence while minimizing loss and maximizing accuracy. `adagrad` was chosen going through the trial and error phase.

- **Metrics**: `accuracy` Calculates how often predictions equal labels. This metric creates two local variables, total and count that are used to compute the frequency with which `y_pred` matches `y_true`.  

**Source**: 
- [7 tips to choose the best optimizer](https://towardsdatascience.com/7-tips-to-choose-the-best-optimizer-47bb9c1219e) by [Davide Giordano](https://medium.com/@davidegiordano)
- [Impact of Optimizers in Image Classifiers](https://towardsai.net/p/l/impact-of-optimizers-in-image-classifiers)
- [Keras Accuracy Metrics](https://keras.io/api/metrics/accuracy_metrics/#:~:text=metrics.,with%20which%20y_pred%20matches%20y_true%20.)

## Trial and error
Part of the process that lead to the current hyperparameters settings and model architecture is documented in [this file](https://github.com/cla-cif/Cherry-Powdery-Mildew-Detector/blob/main/attachments/trial_and_error.pdf). It compares the outputs and inputs of 5 different models, changing one parameter at a time. 

## The rationale to map the business requirements to the Data Visualizations and ML tasks

### Business Requirement 1: Data Visualization 
>The client is interested in having a study that visually differentiates a cherry leaf affected by powdery mildew from a healthy one.

The study is presented in the dashboard which displays:

-  The difference between an average infected leaf and an average healthy leaf.
-  The "mean" and "standard deviation" images for healthy and powdery mildew infected leaves 
-  Image montage for either infected or healthy leaves.
  
[See hypothesis 1 for more information](#hypothesis-1)

### Business Requirement 2: Classification
>The client is interested in telling whether a given cherry leaf is affected by powdery mildew or not.

The client can upload from the dashboard cherry leaves images in `.jpeg` format up to 200MB obtaining immediate feedback on each leaf. The User Interface of the dashboard with a file uploader widget. The user should upload multiple powdery mildew leaf images. It will display the image and a prediction statement, indicating if the leaf is infected or not with powdery mildew and the probability associated with this statement.

### Business Requirement 3: Report
>The client is interested in obtaining a prediction report of the examined leaves. 

Following each batch of uploaded images a downloadable `.csv` report is available with the predicted status. 

## ML Business Case

### Powdery Mildew classificator
- We want an ML model to predict if a leaf is infected with powdery mildew or not, based on the image database provided by the Farmy & Foods company. The problem can be understood as supervised learning, a two/multi-class, single-label, classification model.
- Our ideal outcome is to provide the farmers a faster and more reliable detector for powdery mildew detection.
- The model success metrics are
    - Accuracy of 87% or above on the test set.
- The model output is defined as a flag, indicating if the leaf has powdery mildew or not and the associated probability of being infected or not. The farmers will take a picture of a leaf and upload it to the App. The prediction is made on the fly (not in batches).
- Heuristics: The current detection method is based on a manual inspection. A farmer spends around 30 minutes in each tree, taking a few samples of tree leaves and verifying visually if the leaf tree is healthy or has powdery mildew. Visual criteria is slow and it leaves room to produce inaccurate diagnostics due to human error. 
- The training data to fit the model come from the leaves database provided by Farmy & Foody company and uploaded on Kaggle. This dataset contains 4208 images of cherry leaves. 

![leaf_detector](https://github.com/cla-cif/Cherry-Powdery-Mildew-Detector/blob/main/readme_images/leaf_detector.png)

## Dashboard Design (Streamlit App User Interface)

### Page 1: Quick Project Summary
- Quick project summary
    - General Information:
        - Powdery mildew is a parasitic fungal disease caused by Podosphaera clandestina in cherry trees. When the fungus begins to take over the plants, a layer of mildew made up of many spores forms across the top of the leaves. The disease is particularly severe on new growth, can slow down the growth of the plant and can infect fruit as well, causing direct crop loss.
        - Visual criteria used to detect infected leaves are light-green, circular lesion on either leaf surface and later on a subtle white cotton-like growth develops in the infected area on either leaf surface and on the fruits thus reducing yield and quality."
- Project Dataset
The available dataset provided by Farmy & Foody contains 4208 featured photos of single cherry leaves against a neutral background. The leaves are either healthy or infested by cherry powdery mildew.
- Business requirements:
    1. The client is interested to have a study to visually differentiate between a parasite-contained and uninfected leaf.
    2. The client is interested in telling whether a given leaf contains a powdery mildew parasite or not.
    3. The client is interested in obtaining a prediction report of the examined leaves. 
- Link to this Readme.md file for additional information about the project. 

### Page 2: leaves Visualizer
It will answer business requirement #1
- Checkbox 1 - Difference between average and variability image
- Checkbox 2 - Differences between average parasitised and average uninfected leaves
- Checkbox 3 - Image Montage
- Link to this Readme.md file for additional information about the project. 

### Page 3: Powdery mildew Detector
- Business requirement #2 and #3 information - "The client is interested in telling whether a given leaf is infected with powdery mildew or not and obtaining a downloadable report of the examined leaves."
- Link to download a set of parasite-contained and uninfected leaf images for live prediction on [Kaggle](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves)
- User Interface with a file uploader widget. The user can upload multiple cherry leaves images. It will display the image, a barplot of the visual representation of the prediction and the prediction statement, indicating if the leaf is infected or not with powdery mildew and the probability associated with this statement.
- Table with the image name and prediction results.
- Download button to download the report in a ```.csv``` format. 
- Link to this Readme.md file for additional information about the project. 
  
### Page 4: Project Hypothesis and Validation
- Block for each project hypothesis including statement, explanation, validation and conclusion. See [Hypothesis and validation](#Hypothesis-and-validation)
- Link to this Readme.md file for additional information about the project. 

### Page 5: ML Performance Metrics
- Label Frequencies for Train, Validation and Test Sets
- Dataset percentage distribution among the three sets
- Model performance - ROC curve
- Model accuracy - Confusion matrix
- Model History - Accuracy and Losses of LSTM Model
- Model evaluation result on Test set

## The process of Cross-industry standard process for data mining
CRISP-DM, which stands for Cross-Industry Standard Process for Data Mining, is an industry-proven way to guide your data mining efforts.

- As a methodology, it includes descriptions of the typical phases of a project, the tasks involved with each phase, and an explanation of the relationships between these tasks.
- As a process model, CRISP-DM provides an overview of the data mining life cycle.

**Source**: [IBM - crisp overview](https://www.ibm.com/docs/it/spss-modeler/saas?topic=dm-crisp-help-overview)

**This process is documented using the Kanban Board provided by GitHub in this repository project section [Powdery Mildew detection project](https://github.com/cla-cif/Cherry-Powdery-Mildew-Detector/projects?query=is%3Aopen)**

A kanban board is an agile project management tool designed to help visualize work, limit work-in-progress, and maximize efficiency (or flow). It can help both agile and DevOps teams establish order in their daily work. Kanban boards use cards, columns, and continuous improvement to help technology and service teams commit to the right amount of work, and get it done!

**Source**: [Atlassian - Kanban boards](https://www.atlassian.com/agile/kanban/boards)

![Kanban main](https://github.com/cla-cif/Cherry-Powdery-Mildew-Detector/blob/main/readme_images/github_kanban_main.png)

The CRISP-DM process is divided in [sprints](https://www.atlassian.com/agile/scrum/sprints#:~:text=What%20are%20sprints%3F,better%20software%20with%20fewer%20headaches.). Each sprint has Epics based on each CRISP-DM task which were subsequently split into task. Each task can be either in the *To Do*, *In progress*, *Review* status as the workflow proceeds and contains in-depth details.

![Kanban detail](https://github.com/cla-cif/Cherry-Powdery-Mildew-Detector/blob/main/readme_images/github_kanban_detail.png)

## Bugs

### Fixed Bug
While determining the right hyperparameters for the model to train properly through a *trial and error* process, the accuracy of the validation set was stuck at 0.50000 and presenting high loss. 

![bug](https://github.com/cla-cif/Cherry-Powdery-Mildew-Detector/blob/main/readme_images/bug.png)

- ##  
     - __Description__ : While determining the right hyperparameters for the model to train properly through a *trial and error* process, the accuracy of the validation set was stuck at 0.50000 and presenting high loss. 
     - __Bug__: Among many reasons that could lead to the model not learning from the dataset, this specific case was due to a mismatch between input labels and expected labels. See [Tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator#flow_from_directory)
     - __Fix/Workaround__: The bug was fixed by changing the ```class_mode``` of the datasets from ```binary``` to ```categorical```. ```class_mode``` determines the type of label arrays that are returned. If the output function of the model is expecting ```categorical``` (2D output), labels must be set accordingly. 

## Unfixed Bug

Images producing false predictions

![bug_leaf](https://github.com/cla-cif/Cherry-Powdery-Mildew-Detector/blob/main/readme_images/bug-leaf.jpeg)

- ##  
     - __Description__ : The above image, despite looking healthy was predicted infected. 
     - __Bug__: The glare results in white pixels wrongly interpreted as powdery mildew infection. The background being the same colour of the leaf could be misleading as the model is not able to clearly detect the leaf shape. 
     - __Fix/Workaround__: The model needs further tuning. 

## Deployment
The project is coded and hosted on GitHub and deployed with [Heroku](https://www.heroku.com/). 

### Creating the Heroku app 
The steps needed to deploy this projects are as follows:

1. Create a `requirement.txt` file in GitHub, for Heroku to read, listing the dependencies the program needs in order to run.
2. Set the `runtime.txt` Python version to a Heroku-20 stack currently supported version.
3. `push` the recent changes to GitHub and go to your [Heroku account page](https://id.heroku.com/login) to create and deploy the app running the project. 
3. Chose "CREATE NEW APP", give it a unique name, and select a geographical region. 
4. Add  `heroku/python` buildpack from the _Settings_ tab.
5. From the _Deploy_ tab, chose GitHub as deployment method, connect to GitHub and select the project's repository. 
6. Select the branch you want to deploy, then click Deploy Branch.
7. Click to "Enable Automatic Deploys " or chose to "Deploy Branch" from the _Manual Deploy_ section. 
8. Wait for the logs to run while the dependencies are installed and the app is being built.
9. The mock terminal is then ready and accessible from a link similar to `https://your-projects-name.herokuapp.com/`
10. If the slug size is too large then add large files not required for the app to the `.slugignore` file.
   
### Forking the Repository

- By forking this GitHub Repository you make a copy of the original repository on our GitHub account to view and/or make changes without affecting the original repository. The steps to fork the repository are as follows:
    - Locate the [GitHub Repository](https://github.com/cla-cif/Cherry-Powdery-Mildew-Detector) of this project and log into your GitHub account. 
    - Click on the "Fork" button, on the top right of the page, just above the "Settings". 
    - Decide where to fork the repository (your account for instance)
    - You now have a copy of the original repository in your GitHub account.

### Making a local clone

- Cloning a repository pulls down a full copy of all the repository data that GitHub.com has at that point in time, including all versions of every file and folder for the project. The steps to clone a repository are as follows:
    - Locate the [GitHub Repository](https://github.com/cla-cif/Cherry-Powdery-Mildew-Detector) of this project and log into your GitHub account. 
    - Click on the "Code" button, on the top right of your page.
    - Chose one of the available options: Clone with HTTPS, Open with Git Hub desktop, Download ZIP. 
    - To clone the repository using HTTPS, under "Clone with HTTPS", copy the link.
    - Open Git Bash. [How to download and install](https://phoenixnap.com/kb/how-to-install-git-windows).
    - Chose the location where you want the repository to be created. 
    - Type:
    ```
    $ git clone https://git.heroku.com/cherry-powdery-mildew-detector.git
    ```
    - Press Enter, and wait for the repository to be created.
    - Click [Here](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository#cloning-a-repository-to-github-desktop) for a more detailed explanation. 

__You can find the live link to the site here: [Cherry Powdery Mildew Detector](https://cherry-powdery-mildew-detector.herokuapp.com/)__

## Technologies used

### Platforms
- [Heroku](https://en.wikipedia.org/wiki/Heroku) To deploy this project
- [Jupiter Notebook](https://jupyter.org/) to edit code for this project
- [Kaggle](https://www.kaggle.com/) to download datasets for this project
- [GitHub](https://github.com/): To store the project code after being pushed from Gitpod.
- [Gitpod](https://www.gitpod.io/) Gitpod Dashboard was used to write the code and its terminal to 'commit' to GitHub and 'push' to GitHub Pages.

### Languages
- [Python](https://www.python.org/)
- [Markdown](https://en.wikipedia.org/wiki/Markdown)
  
### Main Data Analysis and Machine Learning Libraries
<pre>
- tensorflow-cpu 2.6.0  used for creating the model
- numpy 1.19.2          used for converting to array 
- scikit-learn 0.24.2   used for evaluating the model
- streamlit 0.85.0      used for creating the dashboard
- pandas 1.1.2          used for creating/saving as dataframe
- matplotlib 3.3.1      used for plotting the sets' distribution
- keras 2.6.0           used for setting model's hyperparamters
- plotly 5.12.0         used for plotting the model's learning curve 
- seaborn 0.11.0        used for plotting the model's confusion matrix
- streamlit             used for creating and sharing this project's interface
</pre>

## Credits

This section lists the sources used to build this project. 

### Content
- The leaves dataset was linked from [Kaggle](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves) and created by [Code Institute](https://www.kaggle.com/codeinstitute)
- The powdery mildew description was taken from [garden design](https://www.gardendesign.com/how-to/powdery-mildew.html) and [almanac](https://www.almanac.com/pest/powdery-mildew)
- The [CRISP DM](https://www.datascience-pm.com/crisp-dm-2/) steps adopted in the [GitHub project](https://github.com/cla-cif/Cherry-Powdery-Mildew-Detector/projects?query=is%3Aopen) were modeled on [Introduction to CRISP-DM](https://www.ibm.com/docs/en/spss-modeler/saas?topic=guide-introduction-crisp-dm) articles from IBM.

### Media
- The banner image is from [shutterstock](https://www.shutterstock.com/image-photo/cherry-tree-green-leaves-isolated-on-120667564), the lettering colour is [Pantone Barbados Cherry](https://www.pantone.com/connect/19-1757-TCX)

### Code
-  This project was developed by Claudia Cifaldi - [cla-cif](https://github.com/cla-cif) on GitHub. 
-  The template used for this project belongs to CodeInstitute - [GitHub](https://github.com/Code-Institute-Submissions) and [website](https://codeinstitute.net/global/).
- App pages for the Streamlit dashboard, data collection and data visualization jupiter notebooks are from [Code Institute WP01](https://github.com/cla-cif/WalkthroughProject01) and where used as a backbone for this project.
- Model learning Curve - C is from [Stack Overflow](https://stackoverflow.com/questions/41908379/keras-plot-training-validation-and-test-set-accuracy) by [Tim Seed](https://stackoverflow.com/users/3257992/tim-seed)
- Classification Report - C is from [Stack Overflow](https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report) by [Özer Özdal](https://stackoverflow.com/users/20249459/%c3%96zer-%c3%96zdal)

### Links to people we like. 

- [GitHub supporting Ukraine](https://github.blog/2022-03-02-our-response-to-the-war-in-ukraine/).
- [GitHub repository by AndrewStetsenko](https://github.com/AndrewStetsenko/Support-Ukraine).

### Acknowledgements

Thank to [Code Institute](https://codeinstitute.net/global/)

### Deployed version at [cherry-powdery-mildew-detector.herokuapp.com](https://cherry-powdery-mildew-detector.herokuapp.com/)
