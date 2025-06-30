import os
import shutil
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

min_images_per_class = 100  # Target number per class
dataset_folder = 'E:/Internship/APPLE FRUIT_DISEASE/AppleDiseaseProject/apple_dataset'
augmented_folder = dataset_folder  # Augment in place

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

for class_name in os.listdir(dataset_folder):
    class_path = os.path.join(dataset_folder, class_name)
    if not os.path.isdir(class_path):
        continue

    images = [img for img in os.listdir(class_path) if img.endswith(('png', 'jpg', 'jpeg'))]
    n_to_generate = max(0, min_images_per_class - len(images))

    if n_to_generate > 0:
        print(f"🔄 Augmenting {class_name} with {n_to_generate} images")
        for i in range(n_to_generate):
            img_name = images[i % len(images)]
            img = Image.open(os.path.join(class_path, img_name))
            x = datagen.random_transform(np.array(img))
            Image.fromarray(x.astype('uint8')).save(os.path.join(class_path, f"aug_{i}.jpg"))

print("✅ Dataset balanced with augmentation")

# --- Step 1–10: Model Training (Same structure as before with improvements) ---
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy

train_dir, val_dir, test_dir = 'train', 'val', 'test'

for d in [train_dir, val_dir, test_dir]:
    if os.path.exists(d): shutil.rmtree(d)
    os.makedirs(d)

for disease in os.listdir(dataset_folder):
    disease_path = os.path.join(dataset_folder, disease)
    if not os.path.isdir(disease_path): continue
    images = os.listdir(disease_path)
    if len(images) < 2:
        print(f"⚠️ Skipping {disease} because it has less than 2 images.")
        continue
    train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)
    for folder, img_list in zip([train_dir, val_dir, test_dir], [train_imgs, val_imgs, test_imgs]):
        os.makedirs(os.path.join(folder, disease), exist_ok=True)
        for img in img_list:
            shutil.copy(os.path.join(disease_path, img), os.path.join(folder, disease, img))

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True)
val_test_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_gen.flow_from_directory(train_dir, target_size=(224,224), batch_size=32, class_mode='categorical')
val_generator = val_test_gen.flow_from_directory(val_dir, target_size=(224,224), batch_size=32, class_mode='categorical')
test_generator = val_test_gen.flow_from_directory(test_dir, target_size=(224,224), batch_size=32, class_mode='categorical', shuffle=False)

# Compute class weights
from sklearn.utils import class_weight
import numpy as np

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes)
class_weights = dict(enumerate(class_weights))

base_model = MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
output = Dense(len(train_generator.class_indices), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss=CategoricalCrossentropy(label_smoothing=0.1),
              metrics=['accuracy'])

es = EarlyStopping(patience=5, restore_best_weights=True)
mc = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')

history = model.fit(train_generator, validation_data=val_generator, epochs=20, callbacks=[es, mc], class_weight=class_weights)

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False
model.compile(optimizer=Adam(learning_rate=1e-5),
              loss=CategoricalCrossentropy(label_smoothing=0.1),
              metrics=['accuracy'])

fine_tune_history = model.fit(train_generator, validation_data=val_generator, epochs=10, callbacks=[es, mc], class_weight=class_weights)

# Plotting
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'] + fine_tune_history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'] + fine_tune_history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy Curve'); plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['loss'] + fine_tune_history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'] + fine_tune_history.history['val_loss'], label='Val Loss')
plt.title('Loss Curve'); plt.legend()
plt.show()

# Evaluation
model = load_model('best_model.h5')
test_loss, test_acc = model.evaluate(test_generator)
print(f"\n✅ Final Test Accuracy: {test_acc*100:.2f}%")

# Confusion Matrix
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
labels = list(test_generator.class_indices.keys())
cm = confusion_matrix(test_generator.classes, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('True')
plt.show()

print(classification_report(test_generator.classes, y_pred, target_names=labels))

# Save Final Model
model.save('E:/Internship/Apple Fruit_Disease/AppleDiseaseProject/fruit_disease_model.h5')
print("✅ Final Model Saved to E:/Internship/Apple Fruit_Disease/AppleDiseaseProject/fruit_disease_model.h5")  
