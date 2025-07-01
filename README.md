# 🍎 Apple Fruit Disease Classification using CNN and Django

This project is a full-stack machine learning web application built to **detect diseases in apple leaves** using a trained **Convolutional Neural Network (CNN)** model and a user-friendly interface developed with **Django**.

Users can upload an image of an apple leaf, and the system will predict whether it’s healthy or affected by one of four common diseases.

---

## 🚀 Features

- 📁 Upload apple leaf images through a web interface
- 🤖 Predict diseases using a CNN model trained with TensorFlow/Keras
- 📊 Displays result with a clean Bootstrap-based UI
- 🧠 Model trained on 5 categories (4 disease types + 1 healthy)
- 📷 Supports real-time user uploads and prediction display

---

## 🧪 Disease Categories

- Apple Scab
- Black Rot
- Cedar Apple Rust
- Bitter Rot
- Healthy

---

## 📂 Project Structure

AppleDiseaseProject/
│
├── manage.py
├── Apple_Disease/ # Main Django project
├── classification/ # App handling views & prediction
│ ├── templates/
│ │ ├── home.html
│ │ └── result.html
│ └── views.py
├── fruit_disease_model.h5 # Trained CNN model
├── static/
│ └── uploads/ # Uploaded leaf images
├── venv/ # Virtual environment (excluded in .gitignore)
└── apple_dataset/ # Training data (not pushed to GitHub)

yaml
Copy
Edit

---

## ⚙️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Pranalikalokhe/Apple-Fruit-Disease-Classification-using-CNN-and-Django.git
   cd Apple-Fruit-Disease-Classification-using-CNN-and-Django
Create & activate virtual environment:


python -m venv venv
.\venv\Scripts\activate
Install dependencies:



Run migrations (if needed):


python manage.py makemigrations
python manage.py migrate
Start the server:


python manage.py runserver
Open in browser:


http://127.0.0.1:8000/
🖼️ Screenshots
![image](https://github.com/user-attachments/assets/c4034129-d73c-49c7-a9bf-ebe0870524cb)


Home Page	Prediction Result

🧠 Technologies Used
Python 3.10

TensorFlow / Keras

Django

Pillow

Bootstrap 5 (UI Styling)

NumPy

👩‍💻 Developed By
Pranali Kalokhe
Data Science Intern, Globeminds Technology Pvt. Ltd.
LinkedIn | GitHub

