import streamlit as st  
import cv2
from PIL import Image, ImageEnhance
import numpy as np 
import pandas as pd 
import os
import keras
from tensorflow.keras.models import load_model
from imutils import paths
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from keras import backend as K
import matplotlib.pyplot as plt


@st.cache(allow_output_mutation=True)
def loadModel():
    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        model = load_model('covid19_model.h5' )#, compile=False)
    print("Model Loaded Succesfully")
    print(model.summary())
    return model

#@st.cache(allow_output_mutation=True)
def Diagnose(image):
    #model,session = model_upload()
    model= loadModel()
    image = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, ( 224, 224))
    cv2.imshow("image", image)
    data = []
    data.append(image)


    # convert the data and labels to NumPy arrays while scaling the pixel
    # intensities to the range [0, 255]
    data = np.array(data) / 255.0
    #K.set_session(session)
    pred = model.predict(data)
    #pred = np.argmax(pred, axis=1)
    #pred = [[0.7,0.9]]
    covid = pred[0][0]
    normal = pred[0][1]
    #st.write(covid, normal)
    data = (np.around([covid, normal],decimals = 2))*100
    covidR = data[0]
    normalR = data[1]
    if covid >= normal:
        st.write("Covid-19 Suspected with {} percent Certainty".format(covidR))
    else:
        st.write("Normal Condition Suspected with {} percent Certainty".format(normalR))
    data = [covidR, normalR]
    my_dict = {"covid":covidR,"normal":normalR}
    df = pd.DataFrame(list(my_dict.items()),columns = ['Status','Percentage']) 
        # Get a color map
    my_cmap = cm.get_cmap('jet')
    
    # Get normalize function (takes data in range [vmin, vmax] -> [0, 1])
    my_norm = Normalize(vmin=0, vmax=100)
    plt.bar("Status", "Percentage", data = df, color = my_cmap(my_norm(data)))
    plt.xlabel("Status")
    plt.ylabel("Percentage")
    plt.title("Percentage of Status")
    st.pyplot()
    

    # st.write(alt.Chart(df).mark_bar().encode(
    # x=alt.X('Status', sort=None),
    # y='Percentage',))

def main():

    """An AI Diagnostic app for detecting Covid-19 from X-ray Scan images"""
    #image
    from PIL import Image
    img = Image.open("Numbers Logo.png")
    st.image(img, width=150,caption="Numb3rs")
    st.title("Numb3rs AI-Based Covid-19 Diagnostic App")
    st.text("Built with Tensorflow, Keras, OpenCv and Streamlit")

    activities = ["upload", "About"]
    choice = st.sidebar.selectbox("Select Activty",activities)

    if choice == 'upload':
        st.subheader("Upload X-ray Image")
        image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])

        if image_file is not None:
            image = Image.open(image_file)
            st.text("X_ray Image")
            is_check = st.checkbox("Display Image")
            if is_check:
                st.image(image,width=300)

        if st.button("Diagnose"):
            Diagnose(image)

    if choice == "About":
        st.subheader("About This Project")
        st.write("This programs uses Deep Transfer learning and X-ray images to detect Covid-19 from chest X-ray radiograph.")
        img2 = Image.open("download.jpg")
        st.text("VGG16 Architecture")
        st.image(img2, width=500,caption="Photo Credit: https://www.researchgate.net/figure/Fig-A1-The-standard-VGG-16-network-architecture-as-proposed-in-32-Note-that-only_fig3_322512435")
    st.write("Powered by Xigma")
        


if __name__ == '__main__':
		main()	