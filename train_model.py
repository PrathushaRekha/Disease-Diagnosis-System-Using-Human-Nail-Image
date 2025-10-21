
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

"""from google.colab import drive
drive.mount('/content/drive')"""

imageSize = [224, 224]

trainPath = r'C:\Users\NANDINI\OneDrive\Desktop\nail\dataset\train'
testPath = r'C:\Users\NANDINI\OneDrive\Desktop\nail\dataset\test'

vgg = VGG16(input_shape=imageSize + [3], weights='imagenet',include_top=False)

for layer in vgg.layers:
  layer.trainable = False
  print(layer)

# our layers - you can add more if you want
x = Flatten()(vgg.output)

prediction = Dense(17, activation='softmax')(x)

# create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy'], run_eagerly=True
)

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(trainPath,
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(testPath,
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

training_set.class_indices

len(training_set)

import sys
# fit the model
r = model.fit(
  training_set,
  validation_data=test_set,
  epochs=5,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set))

#save the model
model.save('vgg-16-nail-disease.h5')

#import load_model class for loading h5 file
from tensorflow.keras.models import load_model
#import image class to process the images
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

#load saved model file
model=load_model('vgg-16-nail-disease.h5')

#load one random image from local system
img = image.load_img('C:/Users/NANDINI/OneDrive/Desktop/nail/dataset/test/clubbing/10.PNG', target_size=(224, 224))

#convert image to array format
x = image.img_to_array(img)

import matplotlib.pyplot as plt

x.shape

x = np.expand_dims(x,axis=0)
img_data=preprocess_input(x)
print(img_data.shape)

model.predict(img_data)

output=np.argmax(model.predict(img_data), axis=1)

# Evaluate the trained VGG16 nail disease model
test_loss, test_accuracy = model.evaluate(test_set)

# Display the results
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

import numpy as np

output=np.argmax([[2.4984448e-04, 7.0662162e-04, 2.1682821e-04, 1.6972199e-04, 9.9644611e-05, 9.8324293e-01, 6.6723401e-04, 5.1942193e-03, 4.8860402e-05, 2.2987171e-05, 2.5026349e-03, 3.5574668e-04, 9.6228459e-06, 4.4953203e-04, 1.4495164e-04, 5.1868602e-04, 5.3999224e-03]], axis=1)

print(output)

index=['Darier_s disease', 'Muehrck-e_s lines', 'aloperia areata', 'beau_s lines', 'bluish nail',
       'clubbing','eczema','half and half nailes (Lindsay_s nails)','koilonychia','leukonychia',
       'onycholycis','pale nail','red lunula','splinter hemmorrage','terry_s nail','white nail','yellow nails']
result = str(index[output[0]])
result

# @title Default title text
import matplotlib.pyplot as plt

# Plot training and validation accuracy
plt.plot(r.history['accuracy'], label='Training Accuracy')
plt.plot(r.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss

plt.plot(r.history['loss'], label='Training Loss')
plt.plot(r.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

import json

# Save the trained model
model.save('vgg16-nail-disease.h5')

# Save class indices for reference during prediction
class_indices = training_set.class_indices
with open('class_indices.json', 'w') as f:
    json.dump(class_indices, f)

print("Model and class indices have been saved successfully.")
