import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import numpy as np

# Load preprocessed data
X_train = np.load('X_train.npy')
X_val = np.load('X_val.npy')
y_train = np.load('y_train.npy')
y_val = np.load('y_val.npy')

# Normalize images
X_train = X_train / 255.0
X_val = X_val / 255.0

IMG_SHAPE = (224, 224, 3)

# Load the base model from MobileNetV2, excluding top layers
base_model = MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base model

# Add custom layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
predictions = Dense(1, activation='sigmoid')(x)  # Binary classification

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Optional: Data augmentation
datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1, horizontal_flip=True)
train_generator = datagen.flow(X_train, y_train, batch_size=16)

# **Early Stopping Callback**
early_stopping = EarlyStopping(
    monitor='val_loss',  # Stop training when validation loss stops improving
    patience=3,          # Wait 3 epochs before stopping
    restore_best_weights=True  # Restore best weights when stopping
)

# Train the model with Early Stopping
model.fit(train_generator, validation_data=(X_val, y_val), epochs=10, callbacks=[early_stopping])

# Save the model for later inference
model.save('fall_detection_model.h5')
print("Model trained and saved as fall_detection_model.h5")
