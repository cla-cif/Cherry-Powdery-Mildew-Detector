import streamlit as st
import matplotlib.pyplot as plt


def page_summary_body():

    st.write("### Quick Project Summary")

    st.info(
        f"**General Information**\n\n"
        f"Powdery mildew is a parasitic fungal disease caused by Podosphaera clandestina in cherry trees."
        f" When the fungus begins to take over the plants, a layer of mildew made up of many spores forms across the top of the leaves.\n\n"
        f" The disease is particularly severe on new growth, can slow down the growth of the plant and can infect fruit as well, causing direct crop loss.\n\n"
        f" Several leaves, infected and healthy, were picked and examined."
        f"\nVisual criteria used to detect infected leaves are:\n\n"
        f"* Light-green, circular lesion on either leaf surface and later on\n "
        f"* a subtle white cotton-like growth develops in the infected area on either leaf surface and ."
        f"on the fruits thus reducing yeld and quality."
        f" \n\n")

    st.warning(
        f"**Project Dataset**\n\n"
        f"The available dataset contains 2104 healthy leaves and 2104 affected leaves "
        f"individually photographed against a neutral background."
        f"")

    st.success(
        f"The project has three business requirements:\n\n"
        f"1 - A study to visually differentiate a healthy from an infected leaf.\n\n"
        f"2 - An accurate prediction whether a given leaf is infected by powdery mildew or not. \n\n"
        f"3 - Download a prediction report of the examined leaves."
        )

    st.write(
        f"For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/cla-cif/Detection-Cherry-Powdery-Mildew#readme).")