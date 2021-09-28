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

COVID_PATH = '/home/binh/dataset/COVID-19_Radiography_Dataset/COVID/'
NORMAL_PATH = '/home/binh/dataset/COVID-19_Radiography_Dataset/Normal/'
PNEUMONIA_PATH = '/home/binh/dataset/COVID-19_Radiography_Dataset/Viral Pneumonia/'

def img_preprocessing(image_path):
    img = cv2.imread(image_path, 0)
    org_img = img.copy()
    brightest = np.max(img)
    darkest = np.min(img)
    T = darkest + 0.9*(brightest - darkest)
    thre_img = cv2.threshold(img, T, 255, cv2.THRESH_BINARY)
    thre_img = thre_img[1]
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.erode(thre_img, kernel, iterations = 5)
    cleaned = cv2.dilate(cleaned, kernel, iterations = 5)
    cleaned = cleaned//255
    img = img * cleaned
    img = org_img - img
    dim = (224, 224)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    B = cv2.bilateralFilter(img, 9, 75, 75)
    R = cv2.equalizeHist(img)
    new_img = cv2.merge((B, img, R))
    return new_img

base_dir = '/home/binh/covid-chestxray-dataset/base_dir'
os.mkdir(base_dir)
train_dir = os.path.join(base_dir, 'train_dir')
os.mkdir(train_dir)
val_dir = os.path.join(base_dir, 'val_dir')
os.mkdir(val_dir)
test_dir = os.path.join(base_dir, 'test_dir')
os.mkdir(test_dir)
Normal = os.path.join(train_dir, 'Normal')
os.mkdir(Normal)
COVID = os.path.join(train_dir, 'COVID')
os.mkdir(COVID)
Viral_Pneumonia = os.path.join(train_dir, 'Viral_Pneumonia')
os.mkdir(Viral_Pneumonia)
Normal = os.path.join(val_dir, 'Normal')
os.mkdir(Normal)
COVID = os.path.join(val_dir, 'COVID')
os.mkdir(COVID)
Viral_Pneumonia = os.path.join(val_dir, 'Viral_Pneumonia')
os.mkdir(Viral_Pneumonia)
Normal = os.path.join(test_dir, 'Normal')
os.mkdir(Normal)
COVID = os.path.join(test_dir, 'COVID')
os.mkdir(COVID)
Viral_Pneumonia = os.path.join(test_dir, 'Viral_Pneumonia')
os.mkdir(Viral_Pneumonia)

folder_1 = os.listdir(COVID_PATH)
folder_1 = shuffle(folder_1)
folder_2 = os.listdir(NORMAL_PATH)
folder_2 = shuffle(folder_2)
folder_3 = os.listdir(PNEUMONIA_PATH)
folder_3 = shuffle(folder_3)
print(len(folder_1))
print(len(folder_2))
print(len(folder_3))