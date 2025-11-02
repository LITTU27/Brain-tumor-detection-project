#!/usr/bin/env python
import os
from flask import Flask, request, render_template
from io import BytesIO
from PIL import Image, ImageOps
import base64
import urllib
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your trained model once at startup
MODEL_PATH = 'save.h5'
model = load_model(MODEL_PATH)


@app.route("/")
@app.route("/first")
def first():
    return render_template('first.html')

@app.route("/login")
def login():
    return render_template('login.html')

@app.route("/chart")
def chart():
    return render_template('chart.html')

@app.route("/performance")
def performance():
    return render_template('performance.html')

@app.route("/index", methods=['GET'])
def index():
    return render_template('index.html')

@app.route("/upload", methods=['POST'])
def upload_file():
    print("Upload endpoint hit")
    try:
        img_file = request.files.get('imagefile')
        if img_file is None or img_file.filename == '':
            error_msg = "Please choose an image file!"
            return render_template('result.html', error_msg=error_msg)
        
        img = Image.open(BytesIO(img_file.read())).convert('RGB')
        img = ImageOps.fit(img, (224, 224), Image.LANCZOS)
    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        return render_template('result.html', error_msg=error_msg)

    # Predict
    out_pred, out_prob = predict(img)
    out_prob_percent = out_prob * 100

    print(f"Prediction: {out_pred}, Confidence: {out_prob_percent:.2f}%")

    # Determine danger class for bootstrap alert style
    # Adjust this depending on your exact output labels
    danger = "danger"
    if out_pred == "Result: Normal":
        danger = "success"
    elif out_pred == "Result: Brain Tumor":
        danger = "danger"
    else:
        danger = "warning"

    # Convert image to base64 for display
    img_io = BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    png_output = base64.b64encode(img_io.getvalue()).decode('utf-8')

    return render_template('result.html', out_pred=out_pred, out_prob=out_prob_percent, danger=danger, processed_file=png_output)


def predict(img):
    # Preprocess image for model
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)

    pred_class = np.argmax(pred, axis=1)[0]
    confidence = float(np.max(pred))

    if pred_class == 0:
        out_pred = "Result: Brain Tumor"
    elif pred_class == 1:
        out_pred = "Result: Normal"
    else:
        out_pred = "Result: Unknown"

    return out_pred, confidence


if __name__ == '__main__':
    app.run(debug=True)
