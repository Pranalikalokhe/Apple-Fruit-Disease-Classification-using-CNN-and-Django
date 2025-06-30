import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, save_img
import numpy as np
from tqdm import tqdm

# Set dataset path and target image count per class
dataset_dir = 'E:/Internship/APPLE FRUIT_DISEASE/AppleDiseaseProject/apple_dataset'
target_count = 100  # desired number of images per class
img_size = (224, 224)

# Augmentation setup
augmenter = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Loop through each class folder
for class_name in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    current_count = len(images)

    if current_count >= target_count:
        print(f"✅ {class_name} has {current_count} images. No augmentation needed.")
        continue

    print(f"🔁 Augmenting '{class_name}' from {current_count} to {target_count} images...")
    augment_needed = target_count - current_count
    image_index = 0
    augmented = 0

    while augmented < augment_needed:
        img_path = os.path.join(class_path, images[image_index % current_count])
        image = load_img(img_path, target_size=img_size)
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, 0)

        aug_iter = augmenter.flow(image_array, batch_size=1)

        # Generate 1 augmented image at a time
        aug_img = next(aug_iter)[0].astype('uint8')
        save_path = os.path.join(class_path, f"aug_{augmented}.jpg")
        save_img(save_path, aug_img)
        augmented += 1
        image_index += 1

    print(f"✅ Finished: '{class_name}' now has {target_count} images.")
