import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
from src.machine_learning.evaluate_clf import load_test_evaluation


def page_ml_performance_metrics():
    version = 'v1'

    st.write("### Images distribution per set and label ")

    labels_distribution = plt.imread(f"outputs/{version}/number_leaves_sets.png")
    st.image(labels_distribution, caption='Labels Distribution on Train, Validation and Test Sets')

    labels_distribution = plt.imread(f"outputs/{version}/sets_distribution_pie.png")
    st.image(labels_distribution, caption='Sets distribution')

    st.warning(
        f"The leaves dataset was divided into three subsets."
        f"Train set (70% of the whole dataset) is the initial data used to 'fit' the model which will learn on this set how to generalize and make prediction on new unseen data."
        f"Validation set (10% of the dataset) helps to improve the model performance by fine-tuning the model after each epoch (one complete pass of the training set through the model)."
        f"The test set (20% of the dataset) informs us about the final accuracy of the model after completing the training phase. It's a batch of data the model has never seen.")
    st.write("---")

    st.write("### Model Performance")

    model_roc = plt.imread(f"outputs/{version}/roccurve.png")
    st.image(model_roc, caption='ROC Curve')

    st.warning(
        f"**ROC Curve**\n\n"
        f"ROC curve is a performance measurement."
        f"It tells how much the model is capable of distinguishing between classes by making accurate predictions."
        f"A ROC curve is constructed by plotting the true positive rate (TPR) against the false positive rate (FPR)."
        f"The true positive rate is the proportion of observations that were correctly predicted (the leaf was predicted healty and it is in fact healthy)"
        f"The false positive rate is the proportion of observations that are incorrectly predicted (the leaf was predicted healthy while it's affected instead)")

    model_cm = plt.imread(f"outputs/{version}/confusion_matrix.png")
    st.image(model_cm, caption='Confusion Matrix')

    st.warning(
        f"**Confusion Matrix**\n\n"
        f"Confusion Matrix is a performance measurement for a classifier."
        f"It is a table with 4 different combinations of predicted and actual values."
        f"True Positive / True Negative: the prediction matches the reality."
        f"False Positive / False Negative: the prediction is opposite of reality (the leaf was predicted infected while it's actually healthy) "
        f"A good model is one which has high TP and TN rates, while low FP and FN rates.")

    model_perf = plt.imread(f"outputs/{version}/model_history.png")
    st.image(model_perf, caption='Model Performance')  

    st.warning(
        f"**Model Performance**\n\n"
        f"The Loss is the sum of errors made for each example in training (loss) or validation (val_loss) sets."
        f"Loss value implies how poorly or well a model behaves after each iteration of optimization."
        f"The accuracy is the measure of how accurate your model's prediction (accuracy) is compared to the true data (val_acc)."
        f"When good model performs well on unseen data it means that it's able to generalize and didn't fit too closely to the training dataset.")


    st.write("### Generalised Performance on Test Set")
    st.dataframe(pd.DataFrame(load_test_evaluation(version), index=['Loss', 'Accuracy']))
    
    st.write(
        f"For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/cla-cif/Detection-Cherry-Powdery-Mildew#readme).")
    