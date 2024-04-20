# LUNG NODULE DETECTION PROJECT
This project, developed for the CS-354 course of Computational Intelligence, focuses on detecting lung nodules using various deep learning models such as DenseNet, ResNet, VGG16, and Inception. Through comparative analysis, it has been determined that the ResNet model yields the highest accuracy for the provided dataset, which was obtained from Roboflow. 
## How to run the code
1. Begin by cloning the repository to your local machine.
2. After cloning the repository, execute the classify_data.py file. This script classifies the data based on the labels present in the annotations.coco file within the dataset. Remember to adjust the "url to the folders and files" in the code to match your setup.
3. Run the aug_preprocessing.py file to preprocess and augment the data.
4. Choose the model you want to test on your data. For example, to run the ResNet model, execute the resnet_model.py file. This process generates two files: resnet_model.json and resnet_model.weights.h5. Note that these files are not uploaded due to their large size. Running the model may take some time.
5. There are two ways to make predictions:
    * Prediction Script: Modify the "url to your image" in the prediction.py file to the image you want to predict the lung nodule on, then execute the script to obtain predictions.
    * Web Interface: Modify the model files in lines 10 and 17 of the app.py file to match the files generated from the model you ran. By default, the model used is resnet_model. Run the app.py file  you will get the link in which the website is running. for example: http://127.0.0.1:5000. On the website, you can upload an image to obtain the prediction graph.
6. in the website, you can upload the image to get the prediction graph.
## Note
Ensure that you have the necessary dependencies installed and configured properly before running the code.
