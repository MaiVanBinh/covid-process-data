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
import pickle
import timeit

start = timeit.default_timer()

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

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
