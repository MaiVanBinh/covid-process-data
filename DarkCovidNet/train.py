from fastai.vision import *
import numpy as np
import fastai

path = Path('/home/binh/covid-chestxray-dataset/base_dir/')

np.random.seed(41)
data = ImageDataBunch.from_folder(path, train="train_dir", valid ="val_dir",
        ds_tfms=get_transforms(), size=(256,256), bs=32, num_workers=4).normalize()

print(data.classes)
print(data.c)
print(data.train_ds)
print(data.valid_ds)