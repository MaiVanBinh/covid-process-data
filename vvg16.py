import os
import cv2
import imageio
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras.layers import Activation
import shutil
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.figure_factory as ff

class_list = ['COVID', 'Normal', 'Viral_Pneumonia']

train_path = '/home/binh/covid-chestxray-dataset/base_dir/train_dir'
valid_path = '/home/binh/covid-chestxray-dataset/base_dir/val_dir'
test_path = '/home/binh/covid-chestxray-dataset/base_dir/test_dir'

NUM_AUG_IMAGES_WANTED = 11000

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

datagen = ImageDataGenerator(rescale=1.0/255)

train_gen = datagen.flow_from_directory(train_path,
                                        target_size=(IMAGE_HEIGHT,IMAGE_WIDTH),
                                        batch_size=10,
                                        class_mode='categorical')

val_gen = datagen.flow_from_directory(valid_path,
                                        target_size=(IMAGE_HEIGHT,IMAGE_WIDTH),
                                        batch_size=10,
                                        class_mode='categorical')

test_gen = datagen.flow_from_directory(test_path,
                                        target_size=(IMAGE_HEIGHT,IMAGE_WIDTH),
                                        batch_size=10,
                                        class_mode='categorical', shuffle=False)

kernel_size = (3,3)
pool_size= (2,2)
dropout_conv = 0.2
dropout_dense = 0.2

model = Sequential()

model.add(Conv2D(64, (3, 3), padding="same", input_shape=(224,224,3)))
model.add(Activation("relu"))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(Conv2D(256, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(Conv2D(256, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(512, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(512, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

#Fully conected
model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(3))
model.add(Activation("softmax"))

model.summary()

model.compile(Adam(learning_rate=0.00001), loss='categorical_crossentropy', 
              metrics=['accuracy'])
              
filepath = "/COVID-19-VGG16.h5"

history = model.fit_generator(train_gen, steps_per_epoch=180, validation_data=val_gen,epochs=200, verbose=1)
vgg_16 = '/home/binh/dataset/covid-process-data/VGG-16'
if Path(vgg_16).is_dir():
    shutil.rmtree(vgg_16) 
os.mkdir(vgg_16)

model.save_weights('/home/binh/dataset/covid-process-data/VGG-16/weights.h5')

with open('/home/binh/dataset/covid-process-data/VGG-16/trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
# ax1.plot(history.history['loss'], color='b', label="Training loss")
# ax1.plot(history.history['val_loss'], color='r', label="validation loss")
# ax1.set_xticks(np.arange(1, 200, 1))
# ax1.set_yticks(np.arange(0, 3, 0.5))

# ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
# ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
# ax2.set_xticks(np.arange(1, 200, 1))

# legend = plt.legend(loc='best', shadow=True)
# plt.tight_layout()
# plt.show()

# model.metrics_names

# val_loss, val_acc = \
# model.evaluate_generator(test_gen)

# print('val_loss:', val_loss)
# print('val_acc:', val_acc)

# test_labels = test_gen.classes

# test_labels

# predictions = model.predict(test_gen, verbose=1)

# from sklearn.metrics import confusion_matrix

# predictions.argmax(axis=1)

# cm = confusion_matrix(test_labels, predictions.argmax(axis=1))

# test_gen.class_indices

# import seaborn as sns
# sns.heatmap(cm, annot=True)