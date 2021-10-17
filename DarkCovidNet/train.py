from fastai.vision import *
import numpy as np
import fastai

path = Path('/content/gdrive/MyDrive/DS/COVIDCHESSXRAY/data/')

np.random.seed(41)
data = ImageDataBunch.from_folder(path, train="train", valid ="valid",
        ds_tfms=get_transforms(), size=(256,256), bs=32, num_workers=4).normalize()

print(data.classes)
print(data.c)
print(data.train_ds)
print(data.valid_ds)