import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import pickle

import h5py

import keras

import tensorflow as tf

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

class load_models:

    def __init__(self, model_path):
        self.model_path = model_path

    def load_feature_extraction_model(self):
        image_feature_model = keras.models.load_model(self.model_path)

        return image_feature_model

obj = load_models("/Users/rahmonolusegunadeniji/Documents/Project/image_models/merged_model2.h5")
m_model = obj.load_feature_extraction_model()

print(m_model.output.shape)
print(m_model.summary())


