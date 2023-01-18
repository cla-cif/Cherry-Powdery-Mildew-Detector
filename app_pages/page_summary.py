import streamlit as st
import matplotlib.pyplot as plt


def page_summary_body():

    st.write("### Quick Project Summary")

    st.info(
        f"**General Information**\n"
        f"* Cherry powdery mildew is a fungal disease caused by Podosphaera clandestina. "
        f" The disease is particularly severe on new growth, and can infect fruit as well, causing direct crop loss"
        f"* Several leaves, infected and healthy, were picked and examined "
        f"Visual criteria used to detect infected leaves are:\n"
        f"light-green, circular lesion on either leaf surface and later on "
        f" "
        f" "
        f" "
        f" \n\n"
        f"**Project Dataset**\n"
        f"* The available dataset contains 2104 healthy leaves and 2104 affected leaves "
        f"individually photographed against a neutral background"
        f"")

    st.write(
        f"* For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/cla-cif/Detection-Cherry-Powdery-Mildew#readme).")
    

    st.success(
        f"The project has 2 business requirements:\n"
        f"* 1 - The client is interested in having a study to visually differentiate "
        f"a healthy from an infected leaf.\n"
        f"* 2 - The client is interested in telling whether a given leaf is infected by powdery mildew or not. "
        )
