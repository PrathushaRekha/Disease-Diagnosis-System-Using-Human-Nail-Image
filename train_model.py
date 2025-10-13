# train_model.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam

# ======================
# 1️⃣ Paths for dataset
# ======================
train_dir = 'dataset/train'
test_dir = 'dataset/test'

# ======================
# 2️⃣ Image preprocessing
# ======================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical'
)

# ======================
# 3️⃣ Load base model (VGG16)
# ======================
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# ======================
# 4️⃣ Build model
# ======================
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes: healthy, fungal, psoriasis
])

# ======================
# 5️⃣ Compile model
# ======================
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ======================
# 6️⃣ Train model
# ======================
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=10
)

# ======================
# 7️⃣ Save model
# ======================
model.save('Vgg-16-nail-disease.h5')
print("✅ Model saved successfully as 'Vgg-16-nail-disease.h5'")
