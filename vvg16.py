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

for item in class_list:
    aug_dir = '/home/binh/covid-chestxray-dataset/aug_dir'
    os.mkdir(aug_dir)
    img_dir = os.path.join(aug_dir, 'img_dir')
    os.mkdir(img_dir)
    img_class = item
    img_list = os.listdir('./home/binh/covid-chestxray-dataset/base_dir/train_dir/' + img_class)
    for fname in img_list:
            src = os.path.join('./home/binh/covid-chestxray-dataset/base_dir/train_dir/' + img_class, fname)
            dst = os.path.join(img_dir, fname)
            shutil.copyfile(src, dst)
    path = aug_dir
    save_path = './home/binh/covid-chestxray-dataset/base_dir/train_dir/' + img_class

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')

    batch_size = 50

    aug_datagen = datagen.flow_from_directory(path,save_to_dir=save_path,save_format='png',target_size=(IMAGE_HEIGHT,IMAGE_WIDTH),batch_size=batch_size)
    num_files = len(os.listdir(img_dir))
    num_batches = int(np.ceil((NUM_AUG_IMAGES_WANTED-num_files)/batch_size))

    for i in range(0,num_batches):
        imgs, labels = next(aug_datagen)
    shutil.rmtree('aug_dir')

print(len(os.listdir('base_dir/train_dir/Normal')))
print(len(os.listdir('base_dir/val_dir/Normal')))
print(len(os.listdir('base_dir/train_dir/COVID')))
print(len(os.listdir('base_dir/val_dir/COVID')))
print(len(os.listdir('base_dir/train_dir/Viral_Pneumonia')))
print(len(os.listdir('base_dir/val_dir/Viral_Pneumonia')))

train_path = 'base_dir/train_dir'
valid_path = 'base_dir/val_dir'
test_path = 'base_dir/test_di

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

aug_dir_1 = 'COVID-19'
os.mkdir(aug_dir_1)
filepath = "./COVID-19-VGG16.h5"

history = model.fit_generator(train_gen, steps_per_epoch=180, validation_data=val_gen,epochs=200, verbose=1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, 200, 1))
ax1.set_yticks(np.arange(0, 3, 0.5))

ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, 200, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()

model.metrics_names

val_loss, val_acc = \
model.evaluate_generator(test_gen)

print('val_loss:', val_loss)
print('val_acc:', val_acc)

test_labels = test_gen.classes

test_labels

predictions = model.predict(test_gen, verbose=1)

from sklearn.metrics import confusion_matrix

predictions.argmax(axis=1)

cm = confusion_matrix(test_labels, predictions.argmax(axis=1))

test_gen.class_indices

import seaborn as sns
sns.heatmap(cm, annot=True)

