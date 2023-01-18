Dataset Content
The dataset contains 5643 images taken from a blood smear workflow (where a drop of blood is placed on a glass slide). The cells are either malaria-parasitised or uninfected.

Business Requirements
As a Data Analyst from Code Institute Consulting, you are requested by the Health Care division to provide actionable insights and data-driven recommendations to a Global Hospital Institution. The client is currently facing challenges in disease detection, especially with malaria.

1 - The client is interested in having a study that visually differentiates a parasitised from an uninfected cell.
2 - The client is interested in telling whether a given cell contains a malaria parasite or not.
Hypothesis and how to validate?
We suspect malaria-parasitised cells have clear marks/signs, typically in the middle of the cell, differentiating them from uninfected cells.
An average image study can help to investigate it
The rationale to map the business requirements to the Data Visualizations and ML tasks
Business Requirement 1: Data Visualization

We will display the "mean" and "standard deviation" images for parasitised and uninfected cells.
We will display the difference between an average parasitised cell and an average uninfected cell.
We will display an image montage for either parasitised or uninfected cells.
Business Requirement 2: Classification

We want to predict if a given cell is infected, or not, with malaria.
We want to build a binary classifier and generate reports.
ML Business Case
MalariaClf
We want an ML model to predict if a cell is infected with malaria or not, based on historical image data. It is a supervised model, a 2-class, single-label, classification model.
Our ideal outcome is to provide the medical team a faster and more reliable diagnostic for malaria detection.
The model success metrics are
Accuracy of 65% or above on the test set.
The model output is defined as a flag, indicating if the cell has malaria or not and the associated probability of being infected or not. The medical staff will do the blood smear workflow as usual and upload the picture to the App. The prediction is made on the fly (not in batches).
Heuristics: The current diagnostic needs an experienced staff and detailed inspection to distinguish infected and not infected cells. A blood smear sample is collected, mixed with a reagent and examined under the microscope. Visual criteria are used to detect malaria parasites. It leaves room to produce inaccurate diagnostics due to human error. On top of that, some specific hospital facilities with malaria centres need more, trained staff and expertise and are typically understaffed.
The training data to fit the model come from the National Institutes of Health (NIH) Website. This dataset contains about 26+ thousand images. We have extracted a subset of 5643 images from this dataset and saved it to Kaggle dataset directory for quicker model training.
Train data - target: infected or not; features: all images
Dashboard Design (Streamlit App User Interface)
Page 1: Quick Project Summary
Quick project summary
General Information
Malaria is a parasitic infection transmitted by the bite of infected female Anopheles mosquitoes.
A blood smear sample is collected, mixed with a reagent and examined in the microscope. Visual criteria are used to detect malaria parasites.
According to WHO, in 2019, there were an estimated 229 million cases of malaria worldwide and an estimated 409 thousand deaths due to this disease. Children <5 years are the most vulnerable group, accounting for 67% (274 thousand) of all malaria deaths worldwide in 2019.
Project Dataset
The available dataset contains 5643 out of +27 thousand images taken from blood smear workflow (when a drop of blood it taken on a glass slide) of cells that are parasitised or uninfected with malaria.
Link to additional information
Business requirements
The client is interested to have a study to visually differentiate between a parasite-contained and uninfected cell.
The client is interested in telling whether a given cell contains a malaria parasite or not.
Page 2: Cells Visualizer
It will answer business requirement 1
Checkbox 1 - Difference between average and variability image
Checkbox 2 - Differences between average parasitised and average uninfected cells
Checkbox 3 - Image Montage
Page 3: Malaria Detector
Business requirement 2 information - "The client is interested in telling whether a given cell contains a malaria parasite or not."
Link to download a set of parasite-contained and uninfected cell images for live prediction.
User Interface with a file uploader widget. The user should upload multiple malaria cell images. It will display the image and a prediction statement, indicating if the cell is infected or not with malaria and the probability associated with this statement.
Table with the image name and prediction results.
Download button to download table.
Page 4: Project Hypothesis and Validation
Block for each project hypothesis, describe the conclusion and how you validated it.
Page 5: ML Performance Metrics
Label Frequencies for Train, Validation and Test Sets
Model History - Accuracy and Losses
Model evaluation result