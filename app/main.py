import os
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the saved model
model = tf.keras.models.load_model('fruits8.keras')

# Image preprocessing
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Class labels
labels = {
    0: 'Apple',
    1: 'Banana',
    2: 'Grape',
    3: 'Mango',
    4: 'Strawberry',
    5: 'chickoo fruit',
    6: 'kiwi fruit',
    7: 'orange fruit'
}

# Upload folder
UPLOAD_FOLDER = 'temp'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Preprocess and predict
        processed_image = preprocess_image(file_path)
        prediction = model.predict(processed_image)[0]

        confidence = float(np.max(prediction))
        predicted_class_index = int(np.argmax(prediction))

        print(f"Prediction vector: {prediction}")
        print(f"Confidence: {confidence}")
        print(f"Predicted index: {predicted_class_index}")

        # Confidence threshold
        THRESHOLD = 0.85

        if confidence < THRESHOLD:
            predicted_class = "Unknown Fruit"
            calory = "N/A"
        else:
            predicted_class = labels.get(predicted_class_index, "Unknown Fruit")
            location = float(request.form.get('gram', 1.0))

            # Estimate calories
            calory_map = {
                'Apple': 52,
                'Banana': 89,
                'Grape': 67,
                'Mango': 60,
                'Strawberry': 33,
                'chickoo fruit': 83,
                'kiwi fruit': 61,
                'orange fruit': 47
            }
            calory = round((location * calory_map.get(predicted_class, 0)) / 100, 2)

        return render_template(
            'index.html',
            prediction=predicted_class,
            calory=calory,
            confidence=round(confidence * 100, 2)
        )

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
