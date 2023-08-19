import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import pickle

import h5py

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from src.utils import m_model


import streamlit as st




def predict_pipepline(Dx, Dx_HPV, STDs_HPV, Hormonal_contraceptives_years, IUD, First_sexual_Intercourse, cigarette_pack_per_year):

    transformer_object_import = open("/Users/rahmonolusegunadeniji/Documents/Project/model_and_transformer/scaler.pkl", "rb")
    scaler = pickle.load(transformer_object_import)

    model_object_import = open("/Users/rahmonolusegunadeniji/Documents/Project/model_and_transformer/model.pkl", "rb")
    model = pickle.load(model_object_import)

    standardized_values = scaler.transform(np.array([[Dx, Dx_HPV, STDs_HPV, Hormonal_contraceptives_years, IUD, First_sexual_Intercourse, cigarette_pack_per_year]]))

    model_prediction = np.round(100*((model.predict_proba(standardized_values)[0][1])), 2)

    print(model_prediction)

    return model_prediction


def image_predict(image):

    #image_feature_obj = open ("/Users/rahmonolusegunadeniji/Documents/Project/image_models/feature_extractor.pkl", "rb")
    #image_feature_model = pickle.load(image_feature_obj)



    data_size = 1

    batch_size = 1

    image_arr = np.array(Image.open(image))

    image_arr = np.expand_dims(image_arr, axis = 0)

    feature_extract = np.zeros([data_size, 9664], dtype = np.uint8)

    Datagen = ImageDataGenerator()

    Generator = Datagen.flow(image_arr,
        batch_size = batch_size
    )

    batch = next(Generator)

    batch = m_model.predict(batch)
    feature_extract[0:1] = batch


    dictionary_1 = {'Dyskeratotic':3, 
                    'Parabasal':1, 
                    'Metaplastic':4, 
                    'Koilocytotic':0,
       'Superficial_Intermediate':2
       }



    class_label = ['Koilocytotic', 'Parabasal', 'Superficial_Intermediate', 'Dyskeratotic', 'Metaplastic']

    model_import = load_model("/Users/rahmonolusegunadeniji/Documents/Project/Image_cervical_models/best_checkpoint2.h5")


    ypred = model_import.predict(feature_extract)

    ypred_idx = np.argmax(ypred, axis=1)[0]

    ypred_label = class_label[ypred_idx]

    return  ypred_label


def main():
    st.title("Cervical Cancer Prediction")
    st.image("cervical-cancer-image1-streamlit.jpeg")
    

    #Dx = st.text_input("Diagnoses Type")
    #Dx_HPV = st.text_input("Have you been diagnosed with HPV?")
    STDs_HPV = st.selectbox("Has the patient been Diagnosed of HPV resulting from STD before?", options=("Yes", "No"))

    if STDs_HPV == "Yes":
        STDs_HPVs = 1
    else:
        STDs_HPVs = 0
    
    IUD = st.selectbox("Does the patient use Intrauterine Device (IUD)?", options=("Yes", "No"))

    if IUD == "Yes":
        IUDs = 1
    else:
        IUDs = 0
    #IUD = st.text_input("Do use Intrauterine Device (IUD)?")

    #STDs_HPV = st.text_input("Have you been Diagnosed of HPV resulting from STD before?")
    Hormonal_contraceptives_years = st.text_input("For how many years has the patient used Hormonal Contraceptives?")

    First_sexual_Intercourse = st.text_input("At what age did the patient have first sexual Intercourse?")
    cigarette_pack_per_year = st.text_input("How many packs of Cigarette does the patient take per year?")

    image = st.file_uploader("Upload a Cervical histopathological image", type=["png","jpg"], accept_multiple_files=False)

    label = ""

    if image is not None:
        st.image(image, caption = "Your uploded image", use_column_width= True)
        label = image_predict(image)

    

    if label == "Koilocytotic":
        Dx = 1
        Dx_HPV = 1
    
    elif label == "Dyskeratotic":
        Dx = 1
        Dx_HPV = 1
    else:
        Dx = 0
        Dx_HPV = 0
    
    st.write("Predicted Cellular changes: ", label)

    outcome = ""

    if st.button("Check Cervical Cancer Likehood"):
        outcome = predict_pipepline(Dx, Dx_HPV, STDs_HPVs, Hormonal_contraceptives_years, IUDs, First_sexual_Intercourse, cigarette_pack_per_year)
    st.success("Your chances of having cervical cancer is {}%".format(outcome))

    st.write("Please click the button below to see your personlaized ")

    tertiary_prevention = [
        "Surgery",
        "Radiotherapy",
        "Chemotherapy",
        "Palliative care"

    ]

    secondary_prevention = [
        "Patient should screen with high performance test equivalent or better than HPV test",
        "Patient should commence immediate treatment or undergo HPV molecular positive test"
    ]

    primary_prevention = [
        "HPV Vaccination",
        "Health information and warnings about the use of Tobacco",
        "Sex education based on patients' age and culture",
        "Education on how to use condom if patient is sexually active"
    ]

    if st.button("Recommendations"):
        if label == "Koilocytotic" or label == "Dyskeratotic":
            st.write("The patient will need to do undergo either of the following or a combination these:")
            st.write(tertiary_prevention)
        elif STDs_HPV == "Yes":
            st.write("The follwing are highly recommended:")
            st.write(secondary_prevention)
        else:
            st.write("Consider advising the patient to do the following:")
            st.write(primary_prevention)




if __name__ == "__main__":
    main()