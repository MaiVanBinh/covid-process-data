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
from pathlib import Path

class_list = ['COVID', 'Normal', 'Viral_Pneumonia']

for item in class_list:
    aug_dir = '/home/binh/covid-chestxray-dataset/aug_dir'
    if Path(aug_dir).is_dir():
        shutil.rmtree(aug_dir) 
    os.mkdir(aug_dir)
    img_dir = os.path.join(aug_dir, 'img_dir')
    os.mkdir(img_dir)
    img_class = item
    img_list = os.listdir('/home/binh/covid-chestxray-dataset/base_dir/train_dir/' + img_class)
    for fname in img_list:
            src = os.path.join('/home/binh/covid-chestxray-dataset/base_dir/train_dir/' + img_class, fname)
            dst = os.path.join(img_dir, fname)
            shutil.copyfile(src, dst)
    path = aug_dir
    save_path = '/home/binh/covid-chestxray-dataset/base_dir/train_dir/' + img_class

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
    shutil.rmtree('/home/binh/covid-chestxray-dataset/aug_dir')
