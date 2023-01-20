## Dataset Content
The dataset contains 4208 featured photos of single cherry leaves against a neutral background. The leaves are either healthy or infested by cherry powdery mildew a biothropic fungus.

## Business Requirements
We were requested by Mrs. Marianne McGuineys, head of IT and Innoviation at Farmy & Foods a company in the agricultural sector to develop a Machine Learning based system to detect instantly whether a certain cherry tree needs to be treated with a fungicide. The requested system should be capable of detecting instantly, using a tree leaf image, if it is healthy or has a powdery mildew. The system was requested by the Farmy & Food company to automate the detection process conducted manually thus far.

1. The client is interested in conducting a study to visually differentiate a cherry leaf that is healthy from one that contains powdery mildew.
2. The client is interested in predicting if a cherry tree is healthy or contains powdery mildew.

### Hypothesis and how to validate?
Based on the cherry leaves database provided by Farmy & Foods, we suspect fungal infected leaves have clear marks/signs, typically on either leaf surface, differentiating them from uninfected leaves.
An average image study can help to investigate it

### The rationale to map the business requirements to the Data Visualizations and ML tasks

## Business Requirement 1: Data Visualization

We will display the "mean" and "standard deviation" images for healthy and powdery mildew infected leaves.<\br>
We will display the difference between an average infected leaf and an average healthy leaf.
We will display an image montage for either infected or healthy leaves.

## Business Requirement 2: Classification

We want to predict if a given leaf is infected, or not, with powdery mildew.
We want to build a binary classifier and generate reports.

## ML Business Case

### Powdery Mildew classificator
- We want an ML model to predict if a leaf is infected with powdery mildew or not, based on the image database provided by the Farmy & Foods company. It is a supervised model, a 2-class, single-label, classification model.
- Our ideal outcome is to provide the farmers a faster and more reliable detector for powdery mildew detection.
- The model success metrics are
    - Accuracy of 65% or above on the test set.
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
Link to additional information
- Business requirements:
    1. The client is interested to have a study to visually differentiate between a parasite-contained and uninfected leaf.
    2. The client is interested in telling whether a given leaf contains a powdery mildew parasite or not.

### Page 2: leaves Visualizer
It will answer business requirement 1
- Checkbox 1 - Difference between average and variability image
- Checkbox 2 - Differences between average parasitised and average uninfected leaves
- Checkbox 3 - Image Montage

### Page 3: Powdery mildew Detector
- Business requirement 2 information - "The client is interested in telling whether a given leaf contains a powdery mildew parasite or not."
- Link to download a set of parasite-contained and uninfected leaf images for live prediction on [Kaggle](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves)
- User Interface with a file uploader widget. The user should upload multiple powdery mildew leaf images. It will display the image and a prediction statement, indicating if the leaf is infected or not with powdery mildew and the probability associated with this statement.
- Table with the image name and prediction results.
- Download button to download table.

### Page 4: Project Hypothesis and Validation
- Block for each project hypothesis, describe the conclusion and how you validated it.

### Page 5: ML Performance Metrics
- Label Frequencies for Train, Validation and Test Sets
- Model History - Accuracy and Losses
- Model evaluation result