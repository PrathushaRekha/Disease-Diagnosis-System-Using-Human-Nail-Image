# app.py
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# ======================
# 1️⃣ Paths
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Project base folder
MODEL_PATH = os.path.join(BASE_DIR, 'Vgg-16-nail-disease.h5')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ======================
# 2️⃣ Load the trained model
# ======================
model = load_model(MODEL_PATH)

# Class labels
CLASS_NAMES = [
            'Darier_s disease', 'Muehrck-e_s lines', 'aloperia areata',
            'beau_s lines', 'bluish nail', 'clubbing', 'eczema',
            'half and half nailes (Lindsay_s nails)', 'koilonychia', 'leukonychia',
            'onycholycis', 'pale nail', 'red lunula', 'splinter hemmorrage',
            'terry_s nail', 'white nail', 'yellow nails'
        ]

# ======================
# 3️⃣ Routes
# ======================
@app.route('/')
def index():
    return render_template('nailhome.html')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    # Save uploaded image
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make prediction
    preds = model.predict(img_array)
    try:
    # Trying to access index 5 (which doesn't exist)
        predicted_class = CLASS_NAMES[np.argmax(preds)]
    except IndexError:
        print("Error: Tried to access an index that is out of range!")
    
    # print(predicted_class)

    return render_template('nailpred.html', prediction=predicted_class, image_path=os.path.join('static', 'uploads', file.filename))

# ======================
# 4️⃣ Run app
# ======================
if __name__ == '__main__':
    app.run(debug=True)
