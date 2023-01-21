import streamlit as st
import matplotlib.pyplot as plt


def page_project_hypothesis_body():
    st.write("### Project Hypothesis and Validation")

    st.info(
        f"We suspect cherry leaves affected by powdery mildew have clear marks, "
        f"typically the first symptom is a light-green, circular lesion on either leaf surface,"
        f"then a subtle white cotton-like growth develops in the infected area. \n\n"
        f"An Image Montage shows the evident difference between a healthy leaf and an infected one. "
        f"Average Image, Variability Image and Difference between Averages samples did not reveal "
        f"any clear pattern to differentiate one from another."

    )
