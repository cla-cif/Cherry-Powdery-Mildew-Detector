import streamlit as st
import matplotlib.pyplot as plt


def page_project_hypothesis_body():
    st.write("### Hypotesis 1 and validation")

    st.success(
        f"Infected leaves have clear marks differentiating them from the healthy leaves."
    )
    st.info(
        f"We suspect cherry leaves affected by powdery mildew have clear marks, "
        f"typically the first symptom is a light-green, circular lesion on either leaf surface,"
        f" then a subtle white cotton-like growth develops in the infected area."
    )
    st.write("To visualize a thorough investigation of infected and healthy leaves visit the Leaves Visualiser tab.")
     
    st.warning(
        f"The model was able to detect such differences and learn how to differentiate and generalize in order to make accurate predictions."
        f" A good model trains its ability to predict classes on a batch of data without adhering too closely to that set of data."
        f" In this way the model is able to generalize and predict future observation reliably because it didn't 'memorize' the relationships between features and labels"
        f" as seen in the training dataset but the general pattern from feature to labels. "
    )


    st.write("### Hypotesis 2 and validation")

    st.success(
        f"Mathematical formulas comparison: ```softmax``` performs better than ```sigmoid``` as activation function for the CNN output layer. "
    )
    st.info(
        f"Both ```softmax``` and ```sigmoid``` are typically used as functions for binary or multi class classification problems."
        f" How an activation function performs on a model can be evaluated by plotting the model's prediction capacity."
        f" The learning curve shows the accuracy and error rate on the training and validation dataset while the model is training.\n\n"
        f" The model trained using ```softmax``` showed less training/validation sets gap and more"
        f" consistent learning rate after the 5th Epoch compared to the model trained using ```sigmoid```."
    )
    st.warning(
        f"In our case the ```softmax``` function performed better. "
    )
    model_perf_softmax = plt.imread(f"streamlit_images/model_history_rgb_softmax.png")
    st.image(model_perf_softmax, caption='Softmax LSTM Loss/Accuracy performance') 
    model_perf_sigmoid = plt.imread(f"streamlit_images/model_history_sigmoid.png")
    st.image(model_perf_sigmoid, caption='Sigmoid LSTM Loss/Accuracy performance')


    st.write("### Hypotesis 3 and validation")

    st.success(
        f"Converting ```RGB``` images to ```grayscale``` improves image classification performance. "
    )
    st.info(
        f"Color digital images are made of pixels, and pixels are made of combinations of primary colors."
        f" Grayscale images, are black-and-white and each pixel is a single sample representing only an amount of light."
        f" A grayscale image due to its nature conveys less information therefore the model is expected to require"
        f" less computational power to train. \n\n However, feeding a model with an RGB image or convert that image to grayscale "
        f" depends on the nature of the images and the information conveyd by the colour."
    )
    st.warning(
        f"Keeping the colour information performed better. The plot shows lower loss and more consistent accuracy."
        f" The difference between rgb and grayscale images' trainable parameters was so small that brought no significant benefit to the computational cost. "
    )
    model_perf_rgb = plt.imread(f"streamlit_images/model_history_rgb_softmax.png")
    st.image(model_perf_rgb, caption='RGB images LSTM Loss/Accuracy performance')
    model_perf_gray = plt.imread(f"streamlit_images/model_history_gray.png")
    st.image(model_perf_gray, caption='Grayscale images LSTM Loss/Accuracy performance') 

    st.write(
        f"For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/cla-cif/Detection-Cherry-Powdery-Mildew#readme).")
