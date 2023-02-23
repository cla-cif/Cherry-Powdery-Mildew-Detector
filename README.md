![Banner]()

## Table of Contents
1. [Dataset Content](#dataset-content)
2. [Business Requirements](#business-requirements)
3. [Hypotesis and validation](#hypotesis-and-validation)
4. [Implementation of the Business Requirements](#the-rationale-to-map-the-business-requirements-to-the-data-visualizations-and-ml-tasks)
5. [ML Business case](#ml-business-case)
6. [Dashboard design](#dashboard-design-streamlit-app-user-interface)
7. [CRISP DM Process](#the-process-of-cross-industry-standard-process-for-data-mining)
8. [Bugs](#bugs)
9. [Deployment](#deployment)
10. [Technologies used](#technologies-used)
11. [Credits](#credits)

## Dataset Content

The dataset contains 4208 featured photos of single cherry leaves against a neutral background. The images are taken from the client's crop fields and show leaves that are either healthy or infested by powdery mildew a biothropic fungus. This disease affects many plant species but the client is particularly concerned about their cherry plantation crop since bitter cherries are their flagship product. 

## Business Requirements

We were requested by our client Farmy & Foods a company in the agricultural sector to develop a Machine Learning based system to detect instantly whether a certain cherry tree presents powdery mildew thus needs to be treated with a fungicide. 
The requested system should be capable of detecting instantly, using a tree leaf image, whether it is healthy or needs attention. 
The system was requested by the Farmy & Food company to automate the detection process conducted manually thus far. The company has thousands of cherry trees, located on multiple farms across the country. As a result, this manual process is not scalable due to the time spent in the manual process inspection.
Link to the wiki section of this repo for the full [business interview](https://github.com/cla-cif/Cherry-Powdery-Mildew-Detector/wiki/Business-understanding-interview). 

Summarizing:

1. The client is interested in conducting a study to visually differentiate a cherry leaf that is healthy from one that contains powdery mildew.
2. The client is interested in predicting if a cherry tree is healthy or contains powdery mildew.
3. The client is interested in obtaining a prediction report of the examined leaves. 

## Hypothesis and validation

1. **Hypotes**: Infected leaves have clear marks differentiating them from the healthy leaves.
   - __How to validate__: Research about the disease and build an average image study can help to investigate it.<br/><br/>

2. **Hypotesis**: Mathematical formulas comparison: ```softmax``` performs better than ```sigmoid``` as activation function for the CNN output layer. 
   - __How to validate__: Understand the kind of problem we are trying to solve and the differences between matemathical functions used to solve that class of problem. Train and compare identical models changing only the activation function of the output layer. <br/><br/>

3. **Hypotesis**: Converting ```RGB``` images to ```grayscale``` improves image classification performance.  
   - __How to validate__: Understand how colours are represented in tensors. Train and compare identical models changing only the image color.

**WHY**A good model trains its ability to predict classes on a batch of data withouth adhering too closely to that set of data. In this way the model is able to generalize and predict future observation reliably because it didn't 'memorize' the relationships between features and labels as seen in the training dataset but the general pattern from feature to labels. 
Understand the concepts of overfitting and underfitting and how to steer away from them. 

### Hypotesis 1
> Infected leaves have clear marks differentiating them from the healthy leaves.

We suspect cherry leaves affected by powdery mildew have clear marks, typically the first symptom is a light-green, circular lesion on either leaf surface, then a subtle white cotton-like growth develops in the infected area. 
An Image Montage shows the evident difference between a healthy leaf and an infected one. 

Difference between average and variability images shows that affected leaves present more white stipes on the center.

![average variability between samples](/workspace/Detection-Cherry-Powdery-Mildew/outputs/v1/avg_var_powdery_mildew.png)
While image difference between average infectead and average infected leaves shows no intuitive difference. 

![average variability between samples](workspace/Detection-Cherry-Powdery-Mildew/outputs/v1/avg_diff.png)

**Sources**:

- [Pacific Nortwest Pest Management Handbooks](https://pnwhandbooks.org/plantdisease/host-disease/cherry-prunus-spp-powdery-mildew)

---
### Hypotesis 2
> Mathematical formulas comparison: ```softmax``` performs better than ```sigmoid``` as activation function for the CNN output layer. 

**1. Introduction**

   1. Understand problem and mathematical functions

First of all let's understand the problem our model is asked to solve. The model is required to assign a cherry leaf one of the two categories: healthy/infected, which makes it a classification problem. It could be seen as a bi
nary classification (healthy vs NOT healthy) or a multiclass classification where each output is assigned one and only one label from more than two classes (just two in our case: healthy vs infected).

If the problem is seen as **binary classification** we will have 1 output node. The probability of the output belonging to one class or the other is within the range of 0 and 1 so if is <0.5 is considered class 0 (healthy) and if >=0.5 is considered class 1 (infected).<br/>
These constraints are given by the ```sigmoid``` function which is also called the _squashing_ function as the classes will converge either to 0 or 1. It's computationally effective but used for binary classification problems only as it suffers major drawbacks which include sharp damp gradients during backpropagation. <br/> 
Backpropagation is where the “learning” or “adjustment” takes place in the neural network in order to adjust the weights of all the nodes throughout the layers of the network. The error value (distance between actual and predicted label) flows back through the network in the opposite direction as before, and it is then used in combination with the derivative of the Sigmoid function. <br/>
The derivative of a function will give us the angle/slope of the graph that the function describes. This value will let the network know weathe to increase or decrease the value of the individual weights in the layers of the network but for a very high or very low value of the error, the derivative of the sigmoid is very low (hence the _squashing_ effect).  

If we see the problem as **multi class classification** we will have 2 output nodes (because I want to predict two classes healthy vs infected). In this case the ```softmax``` function is applied to the output layer. Like the previous case the output of this function lies in the range [0,1] but now we are looking at a probability distribution over the predicted classes which adds up to 1 with the target class having the highes probability. The probability distribution comes from normalizing the output for each class between 0 and 1 and divide by their sum. 

   2. Understand how to evaluate the performance
   
A learning curve is a plot of model learning performance over experience or time.
Learning curves are a widely used diagnostic tool in machine learning for algorithms that learn from a training dataset incrementally. The model can be evaluated on the training dataset and on a hold out validation dataset after each update during training and plots of the measured performance can created to show learning curves.
Reviewing learning curves of models during training can be used to diagnose problems with learning, such as an underfit or overfit model, as well as whether the training and validation datasets are suitably representative. <br/>
Generally, a learning curve is a plot that shows time or experience on the x-axis (Epoch) and learning or improvement on the y-axis (Loss/Accuracy).
   -  **Epoch**: refers to the one entire passing of training data through the algorithm. 
   -  **Loss**: Loss is the penalty for a bad prediction. That is, loss is a number indicating how bad the model's prediction was on a single example. If the model's prediction is perfect, the loss is zero; otherwise, the loss is greater. In our case loss on training set was evaluated against loss on validation set.
   -  **Accuracy**: Accuracy is the fraction of predictions our model got right. Again accuracy on the training swt was measured against accuracy on the validation set. 

In our plot we will be looking for a *good fit* of the learning algorithm which exists between an overfit and underfit model
A good fit is identified by a training and validation loss that decreases to a point of stability with a minimal gap between the two final loss/accuracy values.
We should expect some gap between the train and validation loss/accuracy learning curves. This gap is referred to as the “generalization gap.”
A plot of learning curves shows a good fit if:
   -  The plot of training loss decreases (or increases if it's an accuracy plot) to a point of stability.
   -  The plot of validation loss decreases/increases to a point of stability and has a small gap with the training loss.
   -  Continued training of a good fit will likely lead to an overfit (That's why ML moldels usually have a [early stopping](https://en.wikipedia.org/wiki/Early_stopping) which interrupts the model's learning phase when it stops to improve).
  
**2. Observation**

The model was set to train only on 32 Epoch with no early stoppings, just for the purpose of this hypotesis, and shows overfitting around the 10 last epochs as expected.
The same hyperparameters were set for both examples. 
The model trained using ```softmax``` showed less training/validation sets gap and more consistent learning rate after the 5th Epoch compared to the model trained using ```sigmoid```. 

**3. Conclusion**

In our case the ```softmax``` function performed better. 

**Sources**:
- [Activation Functions Compared With Experiments](https://wandb.ai/shweta/Activation%20Functions/reports/Activation-Functions-Compared-With-Experiments--VmlldzoxMDQwOTQ) by [Sweta Shaw](https://wandb.ai/shweta)
- [Backpropagation in Fully Convolutional Networks](https://towardsdatascience.com/backpropagation-in-fully-convolutional-networks-fcns-1a13b75fb56a#:~:text=Backpropagation%20is%20one%20of%20the,respond%20properly%20to%20future%20urges.) by [Giuseppe Pio Cannata](https://cannydatascience.medium.com/)
- [Understanding The Derivative Of The Sigmoid Function](https://towardsdatascience.com/understanding-the-derivative-of-the-sigmoid-function-cbfd46fb3716#:~:text=The%20Sigmoid%20function%20is%20often,of%20the%20network%20or%20not.) by [Jacob Toftgaard Rasmussen](https://jacobtoftgaardrasmussen.medium.com/)
- [Activation Functions: Comparison of Trends in Practice and Research for Deep Learning](https://arxiv.org/pdf/1811.03378.pdf) by *Chigozie Enyinna Nwankpa, Winifred Ijomah, Anthony Gachagan, and Stephen Marshall*
- [How to use Learning Curves to Diagnose Machine Learning Model Performance](https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/) by [Jason Brownlee](https://machinelearningmastery.com/about)

---
### Hypotesis 3 
> Converting ```RGB``` images to ```grayscale``` improves image classification performance. 

**1. Introduction**

Digital images are made of pixels, every image has three main properties:
   - Size — This is the height and width of an image. It can be represented in centimeters, inches or even in pixels.
   - Color space — Examples are RGB and HSV color spaces.
   - Channel — This is an attribute of the color space. 
  
Each pixel of a coloured image is made of combinations of primary colors represented by a series of code. RGB color space has three types of colors or attributes known as Red, Green and Blue (hence the name RGB).
A grayscale image is one in which the value of each pixel is a single sample representing only an amount of light; that is, it carries only intensity information. Grayscale images, a kind of black-and-white or gray monochrome, are composed exclusively of shades of gray.

In an RGB image where there are three color channels, a pixel value has three numbers, each ranging from 0 to 255 (both inclusive). For example, the number 0 of a pixel in the red channel means that there is no red color in the pixel while the number 255 means that there is 100% red color in the pixel. A single RGB image can be represented using a three-dimensional (3D) NumPy array or a tensor.<br/>
In a grayscale image where there is only one channel, a pixel value has just a single number ranging from 0 to 255 (both inclusive). The pixel value 0 represents black and the pixel value 255 represents white. Therefore a single grayscale image can be represented using a two-dimensional (2D) NumPy array or a tensor because it doesn't need an extra dimension for the color channel. <br/>
Feeding a model with an RGB image or convert that image to grayscale, depends on the nature of the images and the information conveyd by the colour. 
If the color has no significance in the image to classify, indeed a grayscale image requires less computational power to be processed.<br/><br/>

**2. Observation**

The model was set to train only on 32 Epoch with no early stoppings, just for the purpose of this hypotesis, and shows overfitting around the 10 last epochs as expected.
The same hyperparameters were set for both examples. 
The model trained using RGB images showed less training/validation sets gap and more consistent learning rate after the 5th Epoch compared to the model trained using Grayscale images. 
The same CNN applied to an RGB image dataset has 3,715,234 parameters to train compared to 3,714,658 parameters when the same dataset is converted to grayscale. 

   - Comparison of the same image
  
   - Comparison of LSTM 

**3. Conclusion**

Keeping the colour information performed better. The plot shows lower loss and more consistent accuracy. A difference of 676 trainable parameters has no significant benefit on the computational cost. 

Sources:
- [How RGB and Grayscale Images Are Represented in NumPy Arrays](https://towardsdatascience.com/exploring-the-mnist-digits-dataset-7ff62631766a) by [Rukshan Pramoditha](https://rukshanpramoditha.medium.com/)

## The rationale to map the business requirements to the Data Visualizations and ML tasks

### Business Requirement 1: Data Visualization 
>The client is interested in having a study that visually differentiates a cherry leaf affected by powdery mildew from a healthy one.

The study is presented in the dashboard which displays:

-  The difference between an average infected leaf and an average healthy leaf.
-  The "mean" and "standard deviation" images for healthy and powdery mildew infected leaves 
-  Image montage for either infected or healthy leaves.
[See hypotesis 1 for more information])(#Hypotesis 1)
![]()

### Business Requirement 2: Classification
>The client is interested in telling whether a given cherry leaf is affected by powdery mildew or not.

The client can upload from the dashboard cherry leaves images in ```.jpeg``` format up to 200MB obtaining immediate feedback on each leaf. The User Interface of the dashboard with a file uploader widget. The user should upload multiple powdery mildew leaf images. It will display the image and a prediction statement, indicating if the leaf is infected or not with powdery mildew and the probability associated with this statement.

### Business Requirement 3: Report
>The client is interested in obtaining a prediction report of the examined leaves. 

Following each batch of uploaded images a downloadable ```.csv``` report is available with the predicted status. 

## ML Business Case

### Powdery Mildew classificator
- We want an ML model to predict if a leaf is infected with powdery mildew or not, based on the image database provided by the Farmy & Foods company. The problem can be understood as supervised learning, a two/multi-class, single-label, classification model.
- Our ideal outcome is to provide the farmers a faster and more reliable detector for powdery mildew detection.
- The model success metrics are
    - Accuracy of 87% or above on the test set.
- The model output is defined as a flag, indicating if the leaf has powdery mildew or not and the associated probability of being infected or not. The farmers will take a picture of a leaf and upload it to the App. The prediction is made on the fly (not in batches).
- Heuristics: The current detection method is based on a manual inspection. A farmer spends around 30 minutes in each tree, taking a few samples of tree leaves and verifying visually if the leaf tree is healthy or has powdery mildew. Vusual criteria is slow and it leaves room to procduce inaccurate diagnostics due to human error. 
- The training data to fit the model come from the leaves database provided by Farmy & Foody company and uploaded on Kaggle. This dataset contains 4208 images of cherry leaves. 

## Dashboard Design (Streamlit App User Interface)

### Page 1: Quick Project Summary
- Quick project summary
    - General Information:
        - Powdery mildew is a parasitic fungal disease caused by Podosphaera clandestina in cherry trees. When the fungus begins to take over the plants, a layer of mildew made up of many spores forms across the top of the leaves. The disease is particularly severe on new growth, can slow down the growth of the plant and can infect fruit as well, causing direct crop loss.
        - Visual criteria used to detect infected leaves are light-green, circular lesion on either leaf surface and later on a subtle white cotton-like growth develops in the infected area on either leaf surface and on the fruits thus reducing yeld and quality."
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
- Business requirement #2 and #3 information - "The client is interested in telling whether a given leaf is infected with powdery mildew or not and obtaining a donwnloadable report of the examined leaves."
- Link to download a set of parasite-contained and uninfected leaf images for live prediction on [Kaggle](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves)
- User Interface with a file uploader widget. The user can upload multiple cherry leaves images. It will display the image, a barplot of the visual representation of the prediction and the prediction statement, indicating if the leaf is infected or not with powdery mildew and the probability associated with this statement.
- Table with the image name and prediction results.
- Download button to download the report in a ```.csv``` format. 
- Link to this Readme.md file for additional information about the project. 
  
### Page 4: Project Hypothesis and Validation
- Block for each project hypothesis including statement, explaination, validation and conclusion. See [Hypotesis and validation](#Hypothesis-and-validation)
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

Source: [IBM](https://www.ibm.com/docs/it/spss-modeler/saas?topic=dm-crisp-help-overview)

**This process is documented using the Kanban Board provided by GitHub in this repository project section [Powdery Mildew detection project](https://github.com/cla-cif/Cherry-Powdery-Mildew-Detector/projects?query=is%3Aopen)**

A kanban board is an agile project management tool designed to help visualize work, limit work-in-progress, and maximize efficiency (or flow). It can help both agile and DevOps teams establish order in their daily work. Kanban boards use cards, columns, and continuous improvement to help technology and service teams commit to the right amount of work, and get it done!

Source: [Atlassian](https://www.atlassian.com/agile/kanban/boards)

![Kanban main]()

The CRISP-DM process is divided in [sprints](https://www.atlassian.com/agile/scrum/sprints#:~:text=What%20are%20sprints%3F,better%20software%20with%20fewer%20headaches.). Each sprint has Epics based on each CRISP-DM task which were subsequently split into task. Each task can be either in the *To Do*, *In progress*, *Review* status as the workflow proceeeds and contains in-depth details.

![Kanban detail]()

## Fixed Bug
While determining the right hyperparameters for the model to train properly through a *trial and error* process, the accuracy of the validation set was stuck at 0.50000 and presenting high loss. 

![bug]()

The bug was fixed by changing the ```class_mode``` of the datasets from ```binary``` to ```categorical```. ```class_mode``` determins the type of label arrays that are returned. If the output function of the model is expecting ```categorical``` (2D output), labels must be set accordingly. 

Source: [Tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator#flow_from_directory)

## Unfixed Bug

## Deployment

### Heroku
- The App live link is: https://YOUR_APP_NAME.herokuapp.com/
- Set the runtime.txt Python version to a Heroku-20 stack currently supported version.
- The project was deployed to Heroku using the following steps:
  1. Log in to Heroku and create an App
  2. At the Deploy tab, select GitHub as the deployment method.
  3. Select your repository name and click Search. Once it is found, click Connect.
  4. Select the branch you want to deploy, then click Deploy Branch.
  5. The deployment process should happen smoothly if all deployment files are fully functional. Click now the button Open App on the top of the page to access your App.
  6. If the slug size is too large then add large files not required for the app to the .slugignore file.

## Technologies used

### Platforms
- [Jupiter Notebook](https://jupyter.org/)
- [Streamlit](https://streamlit.io/)
- [Kaggle](https://www.kaggle.com/)

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
- matplotlib 3.3.1      used for plotting the sets'distribution
- keras 2.6.0           used for setting model's hyperparamters
- plotly 5.12.0         used for plotting the model's learning curve 
- seaborn 0.11.0        used for plotting the model's confusion matrix
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
- App pages for the Streamlit dashboard, data collection and data visualization jupiter notebooks are from [Code Institute WP01](https://github.com/cla-cif/WalkthroughProject01) and where used as a backbone for this project.
- Model learning Curve - C is from [Stack Overflow](https://stackoverflow.com/questions/41908379/keras-plot-training-validation-and-test-set-accuracy) by [Tim Seed](https://stackoverflow.com/users/3257992/tim-seed)

### Acknowledgements

Thank to [Code Institute](https://codeinstitute.net/global/)

