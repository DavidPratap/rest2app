import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import streamlit as st
# Step1: Load the model
model=load_model("cats_dogs_small_3.h5")

# Step2: Load and View the Image 
uploaded_file=st.file_uploader("choose the database", accept_multiple_files=False)
if uploaded_file is not None:
    file=uploaded_file
else:
    file='image.jpg'
    
if st.checkbox("View Image", False):
    img=Image.open(file)
    st.image(img)
    
# Step3 : Preprocess the image
image=load_img(file, target_size=(150, 150))
image_array=img_to_array(image)
image_array_reshaped=np.expand_dims(image_array, axis=0)

# Step4 : Get the prediction 
prediction=model.predict(image_array_reshaped)[0][0]
pred=int(prediction)
if st.button("predict"):
    if pred==1:
        st.write("Dog")
    else:
        st.write("Cat")


    
