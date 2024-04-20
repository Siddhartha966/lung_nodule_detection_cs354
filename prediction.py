from __future__ import print_function

from sklearn.preprocessing import LabelEncoder , OneHotEncoder
import cv2
import warnings
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing import image
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input



# Load the model architecture from JSON file
with open("vgg_model.json", "r") as json_file:
    loaded_model_json = json_file.read()


# Load the model architecture
loaded_model = keras.models.model_from_json(loaded_model_json)


# Load the model weights
loaded_model.load_weights("vgg_model.weights.h5")


# Now you can use the loaded model for predictions
from matplotlib.pyplot import imread
from matplotlib.pyplot import imshow
img_path = 'E:\CS_354_final_2\classified_images\images_benign\seq1_train_images_1_jpg.rf.e1f6020ec74547c3eefcbcd5acd2d7d1.jpg'
img = image.load_img(img_path, target_size=(300, 300))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print('Input image shape:', x.shape)
my_image = imread(img_path)
imshow(my_image)


# Make predictions using the loaded model
predictions = loaded_model.predict(x)
print(predictions)