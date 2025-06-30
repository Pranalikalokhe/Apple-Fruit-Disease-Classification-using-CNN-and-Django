from django.shortcuts import render, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from django.core.files.storage import FileSystemStorage
import os

# Load the trained model
model = load_model('E:/Internship/Apple Fruit_Disease/AppleDiseaseProject/fruit_disease_model.h5')

# Define class names
class_names = ['Apple__Bitter_Rot', 'Apple___Black_rot', 'Apple___Cedar_rust', 'Apple___healthy', 'Apple___Apple_scab']

def home(request):
    return render(request, 'home.html')


def predict_disease(request):
    if request.method == 'POST':
        image_file = request.FILES.get('image')
        if not image_file:
            return render(request, 'result.html', {'result': 'No image uploaded.'})

        try:
            # Save the uploaded file
            fs = FileSystemStorage(location='static/uploads/')
            filename = fs.save(image_file.name, image_file)
            file_url = fs.url(filename)

            # Preprocess image
            img_path = os.path.join('static/uploads', filename)
            img = Image.open(img_path).resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Make prediction
            prediction = model.predict(img_array)
            predicted_class = class_names[np.argmax(prediction)]

            return render(request, 'result.html', {
                'result': predicted_class,
                'uploaded_image_url': file_url
            })

        except Exception as e:
            return render(request, 'result.html', {'result': f'Error: {str(e)}'})

    return redirect('/')
