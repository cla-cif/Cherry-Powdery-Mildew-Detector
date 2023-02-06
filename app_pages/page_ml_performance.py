import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
from src.machine_learning.evaluate_clf import load_test_evaluation


def page_ml_performance_metrics():
    version = 'v1'

    st.write("### Train, Validation and Test Set: Labels Frequencies")

    labels_distribution = plt.imread(f"outputs/{version}/number_leaves_sets.png")
    st.image(labels_distribution, caption='Labels Distribution on Train, Validation and Test Sets')
    st.write("---")


    st.write("### Model History")

    model_roc = plt.imread(f"outputs/{version}/roccurve.png")
    st.image(model_roc, caption='ROC Curve')

    model_cm = plt.imread(f"outputs/{version}/confusion_matrix.png")
    st.image(model_cm, caption='Confusion Matrix')

    model_perf = plt.imread(f"outputs/{version}/model_training_acc.png")
    st.image(model_perf, caption='Model Performance')  

    st.write("### Generalised Performance on Test Set")
    st.dataframe(pd.DataFrame(load_test_evaluation(version), index=['Loss', 'Accuracy']))
    
    