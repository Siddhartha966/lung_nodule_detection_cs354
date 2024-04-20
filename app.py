from flask import Flask, request, render_template, jsonify
from keras.models import model_from_json
from PIL import Image
import io
import numpy as np

app = Flask(__name__)

# Load the trained model architecture from JSON file
with open("resnet_model.json", "r") as json_file:
    loaded_model_json = json_file.read()

# Load the model architecture
inception_model = model_from_json(loaded_model_json)

# Load model weights
inception_model.load_weights("resnet_model.weights.h5")

# Function to preprocess image
def preprocess_image(img):
    # Preprocess image here according to your model requirements
    # Example: resize image to match model input size, normalize pixel values, etc.
    return img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Receive input image from form
    file = request.files['file']
    
    # Read image as PIL Image
    img = Image.open(io.BytesIO(file.read()))

    # Resize image to 300x300
    img = img.resize((300, 300))

    # Preprocess image for prediction
    img = preprocess_image(img)

    # Convert PIL Image to numpy array
    img_array = np.array(img)

    # Expand dimensions to match model input shape
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction_i = inception_model.predict(img_array)

    # Calculate probabilities for the bar plot
    benign_probability = float(prediction_i[0][0])
    malignant_probability = 1 - benign_probability
    print(benign_probability)
    print(malignant_probability)

    # Return prediction result as JSON
    return jsonify({'benign_probability': benign_probability, 'malignant_probability': malignant_probability})

if __name__ == "__main__":
    app.run()
