from __future__ import print_function

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import cv2
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.applications.resnet import ResNet50, preprocess_input

# Load ResNet50 model pre-trained on ImageNet data
base_model = ResNet50(weights='imagenet', include_top=False)

# Freeze all layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully connected layer
x = Dense(1024, activation='relu')(x)

# Add a logistic layer for the number of classes
predictions = Dense(2, activation='softmax')(x)

# Final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data preparation
ct_scan_dir = 'E:\CS_354_final_2\classified_images'

datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.05)

train_generator = datagen.flow_from_directory(
    ct_scan_dir,
    target_size=(300, 300),
    batch_size=32,
    class_mode='categorical',
    subset='training')

validation_generator = datagen.flow_from_directory(
    ct_scan_dir,
    target_size=(300, 300),
    batch_size=32,
    class_mode='categorical',
    subset='validation')

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size)

# Evaluate the model
loss, accuracy = model.evaluate(validation_generator)
print("Loss:", loss)
print("Accuracy:", accuracy)

# Save the model architecture in JSON format
model_json = model.to_json()
with open("resnet_model.json", "w") as json_file:
    json_file.write(model_json)

# Save the model weights
model.save_weights("resnet_model.weights.h5")
