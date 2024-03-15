import matplotlib.pyplot as plt
from PIL import Image
import io
import cv2
import numpy as np
from tensorflow.keras.models import load_model



uploaded = "3391_left.jpeg"

img = Image.open(uploaded)
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

