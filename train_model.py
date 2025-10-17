from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt

# ===============================
# Configuration
# ===============================

# Image dimensions expected by VGG16
IMAGE_SIZE = [224, 224]

# Paths to dataset (update if needed)
TRAIN_PATH = r'C:\Users\NANDINI\OneDrive\Desktop\nail\dataset\train'
TEST_PATH = r'C:\Users\NANDINI\OneDrive\Desktop\nail\dataset\test'


# ===============================
# Model Setup (VGG16 Base + Custom Head)
# ===============================

# Load the VGG16 model with pretrained ImageNet weights
# Exclude the top (fully-connected) layers
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# Freeze all layers in the VGG16 base to use pretrained features
for layer in vgg.layers:
    layer.trainable = False

# Flatten the output of VGG16 and add a new dense layer for classification
x = Flatten()(vgg.output)
output = Dense(17, activation='softmax')(x)  # 17 output classes for nail diseases

# Build the final model
model = Model(inputs=vgg.input, outputs=output)

# Display model structure
model.summary()

# ===============================
# Compile the Model
# ===============================

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'],
    run_eagerly=True  # Optional for debugging/training clarity
)

# ===============================
# Data Preparation
# ===============================

# Training data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Validation data should only be rescaled
val_datagen = ImageDataGenerator(rescale=1./255)

# Flow images in batches from respective directories
training_set = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_set = val_datagen.flow_from_directory(
    TEST_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# ===============================
# Model Training
# ===============================

history = model.fit(
    training_set,
    validation_data=test_set,
    epochs=5,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)

# ===============================
# Save Trained Model
# ===============================

model.save('/content/Vgg-16-nail-disease.h5')
print("\nModel saved as Vgg-16-nail-disease.h5 in /content directory.")

# ===============================
# (Optional) Plot Training History
# ===============================

plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
