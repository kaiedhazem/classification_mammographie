# les imports
import os
import pandas as pd
import tensorflow as tf
keras = tf.keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import math
import numpy as np

SEED=100
#taille du lot
BATCH_SIZE = 20
data = 'C:/Users/asus/Desktop/IAproject/dataset/'

# les dossiers du training and validation 
training = os.path.join(data, 'train')
validation = os.path.join(data, 'validation')

# classification du training 
train_BENIN = os.path.join(training, 'BENIN')
train_CANCER = os.path.join(training, 'CANCER')
train_NORMAL = os.path.join(training, 'NORMAL')

# classification du training 
validation_BENIN = os.path.join(validation, 'BENIN')
validation_CANCER = os.path.join(validation, 'CANCER')
validation_NORMAL = os.path.join(validation, 'NORMAL')

# Toutes les images seront redimensionnées de 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Images d'entraînement de flux par lots de 20 à l'aide du générateur train_datagen
train_generator = train_datagen.flow_from_directory(
        training,  
        target_size=(128, 128), 
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical')

# Images de validation de flux par lots de 20 à l'aide du générateur test_datagen
generateur_validataion = test_datagen.flow_from_directory(
        validation,
        target_size=(128, 128),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical'
        )

#construction du model
model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(128, 128, 1)),
      tf.keras.layers.MaxPooling2D(2, 2),
      tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(3, activation='softmax')
  ])
#compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#afficher le résumé
model.summary()

#entrianement du modéle (fitting) 
class_weight = {0: 5,
                1: 4,
                2: 2}
history = model.fit(
      train_generator,
      steps_per_epoch=100,  
      epochs=10,
      class_weight=class_weight,
      validation_data=generateur_validataion,
      validation_steps=50,  
      verbose=1)
ypred=model.predict(generateur_validataion)
#print(ypred)
ypred.shape
#extraction des données réel de y
longeur = len(generateur_validataion.filenames)
n = math.ceil(longeur / (1.0 * BATCH_SIZE)) 
yreel = np.empty([longeur, 3])
j=0
for i in range(0,int(n)):
    for item in np.array(generateur_validataion[i][1]):
        yreel[j] = item
        j+=1
#matrice de confusion
matrice = confusion_matrix(yreel.argmax(axis=1), ypred.argmax(axis=1))
print(matrice)
#enregistrement du modéle
model.save("C:/Users/asus/Desktop/IAproject/model")