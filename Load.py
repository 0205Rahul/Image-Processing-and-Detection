import os.path
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load image and resize to (256, 256)
image = load_img('images.jpg', target_size=(256, 256))

# Convert image to NumPy array and preprocess
image_array = img_to_array(image)
normalized_image = image_array / 255.0

# Reshape image to match model input shape
input_image = np.reshape(normalized_image, (1, 256, 256, 3))

# Load model
new_model = load_model(os.path.join('models', 'imageclassifier.h5'))

# Make prediction
yhat = new_model.predict(input_image)

# Check predicted class
if yhat[0][0] > 0.5:
    print("Truck")
else:
    print("Car")
