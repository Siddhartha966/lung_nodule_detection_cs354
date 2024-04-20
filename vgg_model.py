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
# from keras.utils import layer_utils
from keras.utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input

# from keras.applications.imagenet_utils import _obtain_input_shape # this will work for older versions of keras. 2.2.0 or before
# from keras import get_source_inputs


def VGGupdated(input_tensor=None,classes=2):    
   
    img_rows, img_cols = 300, 300   # by default size is 224,224
    img_channels = 3

    img_dim = (img_rows, img_cols, img_channels)
   
    img_input = Input(shape=img_dim)
    
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    # Create model.
   
     
    model = Model(inputs = img_input, outputs = x, name='VGGdemo')


    return model


model = VGGupdated(classes = 2) # malignant and benign

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

ct_scan = os.listdir('E:\CS_354_final_2\classified_images')
ctscanimages = []
for item in ct_scan:
 # Get all the file names
 all_ctscans = os.listdir('E:\CS_354_final_2\classified_images' + '/' +item)


 # Add them to the list
 for room in all_ctscans:
    ctscanimages.append((item, str('E:\CS_354_final_2\classified_images' + '/' +item) + '/' + room))

# Build a dataframe        
ctscanimages_df = pd.DataFrame(data=ctscanimages, columns=['image type', 'image'])

path = 'E:\CS_354_final_2\classified_images/'


im_size = 300

images = []
labels = []

for i in ct_scan:
    data_path = path + str(i)  
    filenames = [i for i in os.listdir(data_path) ]
   
    for f in filenames:
        img = cv2.imread(data_path + '/' + f)
        img = cv2.resize(img, (im_size, im_size))
        images.append(img)
        labels.append(i)


images = np.array(images)

images = images.astype('float32') / 255.0
images.shape 


y=ctscanimages_df['image type'].values
y_labelencoder = LabelEncoder ()
y = y_labelencoder.fit_transform (y)


# Assuming y is a 1D array
y = y.reshape(-1, 1)

# Initialize the OneHotEncoder
onehotencoder = OneHotEncoder()

# Fit and transform the data
Y = onehotencoder.fit_transform(y).toarray()

# Now Y will contain the one-hot encoded values
print(Y.shape)  # Shape of Y


from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
images, Y = shuffle(images, Y, random_state=1)
train_x, test_x, train_y, test_y = train_test_split(images, Y, test_size=0.05, random_state=415)
#inpect the shape of the training and testing.
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)


model.fit(train_x, train_y, epochs = 10, batch_size = 32) 


preds = model.evaluate(test_x, test_y)
print ("Loss = " + str(preds[0]))

# Save the model architecture in JSON format
model_json = model.to_json()
with open("vgg_model.json", "w") as json_file:
    json_file.write(model_json)

# Save the model weights
model.save_weights("vgg_model.weights.h5")

