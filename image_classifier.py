

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.applications import MobileNetV2
from keras import layers, models
import os

# Data directories
train_dir = 'dataset/train'
val_dir = 'dataset/validation'

# Data preprocessing and augmentation
img_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')

val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')

# Model with Transfer Learning
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the base

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(train_generator, epochs=5, validation_data=val_generator)

# Save model
model.save('cat_dog_classifier.h5')
