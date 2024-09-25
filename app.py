from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('cat_dog_classifier.h5')

# Define the upload folder
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for predicting the uploaded image
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocess the uploaded image
        img = image.load_img(filepath, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Make prediction
        prediction = model.predict(img_array)

        # Interpret the result
        if prediction[0][0] > 0.5:
            result = 'Dog'
        else:
            result = 'Cat'

        return render_template('index.html', result=result, img_src=filepath)

if __name__ == '__main__':
    app.run(debug=True)
