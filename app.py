from flask import Flask, render_template,request

import matplotlib.pyplot as plt
from PIL import Image
import io
import cv2
import numpy as np
from tensorflow.keras.models import load_model



app = Flask(__name__)

# Specify the location of the static folder
app.config['STATIC_FOLDER'] = 'static'

def get_hello(img):

    img = Image.open(img)
    img_array = np.array(img)
    img_resized = cv2.resize(img_array, (224, 224))  
    img_resized = img_resized / 255.0  
    img_resized = np.clip(img_resized * 255, 0, 255).astype(np.uint8)

    # plt.imshow(img_resized)  
    # plt.axis('off')
    # plt.show()
    model_path = 'effB3CNNDR.h5'  # Update with your actual folder path
    model = load_model(model_path)

    prediction = model.predict(np.expand_dims(img_resized, axis=0))

    print('Prediction:', prediction)
    prediction = model.predict(np.expand_dims(img, axis=0))
    predicted_class = np.argmax(prediction)
    print('Predicted Stage:', predicted_class)

    class_labels = ['NO_DR', 'Mild', 'Moderate', 'Sever', 'Proleferate']
    predicted_stage = class_labels[predicted_class]
    print('Predicted Stage:', predicted_stage)

    return predicted_stage


# @app.route('/')
# def index():
#     return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    out = None
    uploaded_file = None
    if request.method == 'POST':
        uploaded_file = request.files['file']

    # Call the get_hello function to get the greeting message
    if uploaded_file is not None:
        out = get_hello(img=uploaded_file)

    return render_template('index.html', name=out)


if __name__ == '__main__':
    app.run(debug=True)