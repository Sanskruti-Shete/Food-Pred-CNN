import streamlit as st
import tensorflow as tf
import numpy as np

#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(64,64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Prediction"])
 
#Home Page
if(app_mode=="Home"):
    st.header("FoodLens: Fruits and Veggies Recognition System")
    image_path="home.jpg"
    st.image(image_path)

#About Page
elif(app_mode=="About"):
    st.header("About Project")
    st.subheader("About Dataset")
    st.text("This dataset encompasses images of various fruits and vegetables")
    st.code("Fruits: Banana, Apple, Pear, Grapes, Orange, Kiwi, Watermelon, Pomegranate, Pineapple, Mango")
    st.code("Vegetables: Cucumber, Carrot, Capsicum, Onion, Potato, Lemon, Tomato, Radish, Beetroot, Cabbage, Lettuce, Spinach, Soybean, Cauliflower, Bell Pepper, Chilli Pepper, Turnip, Corn, Sweetcorn, Sweet Potato, Paprika, Jalapeño, Ginger, Garlic, Peas, Eggplant")
    st.subheader("Content")
    st.text("The dataset is organized into three main folders:")
    st.text("1. Train: Contains 100 images per category.")
    st.text("2. Test: Contains 10 images per category.")
    st.text("3. Validation: Contains 10 images per category.")

#Prediction Page
elif(app_mode=="Prediction"):
    st.header("Model Prediction")
    test_image=st.file_uploader("Choose an image:")
    if(st.button("Show Image")):
        st.image(test_image, width=4,use_column_width=True)
    #Predict Button
    if(st.button("Predict")):
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading labels
        with open("labels.txt") as f:
            content = f.readlines()
        label = []
        for i in content:
            label.append(i[:-1])
        st.success("Model is Predicting it's a {}".format(label[result_index]))
