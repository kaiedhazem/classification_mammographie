import os
import pandas as pd
import tensorflow as tf
keras = tf.keras
from keras.preprocessing.image import ImageDataGenerator

SEED=100
BATCH_SIZE = 20
data = 'C:/Users/asus/Desktop/IAproject/dataset/'

# les dossiers du training and validation 
training = os.path.join(data, 'train')
validation = os.path.join(data, 'validation')

# classification du training 
train_BEN = os.path.join(training, 'BEN')
train_CAN = os.path.join(training, 'CAN')
train_NOR = os.path.join(training, 'NOR')

# classification du training 
validation_BEN = os.path.join(validation, 'BEN')
validation_CAN = os.path.join(validation, 'CAN')
validation_NOR = os.path.join(validation, 'NOR')

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        training,  
        target_size=(128, 128), 
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical')

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(
        validation,
        target_size=(128, 128),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical'
        )

#build model
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

#desplay summary
model.summary()

#fitting model 
class_weight = {0: 5,
                1: 5,
                2: 1}
history = model.fit(
      train_generator,
      steps_per_epoch=100,  # 2000 images = batch_size * steps
      epochs=10,
      class_weight=class_weight,
      validation_data=validation_generator,
      validation_steps=50,  # 1000 images = batch_size * steps
      verbose=1)

#save model
model.save('C:/Users/asus/Desktop/IAproject/model')