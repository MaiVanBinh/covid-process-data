from fastai.vision import *
import numpy as np
import fastai
import warnings
warnings.simplefilter("ignore")

path = Path('/home/binh/covid-chestxray-dataset/base_dir/')

np.random.seed(41)
data = ImageDataBunch.from_folder(path, train="train_dir", valid ="val_dir",
        ds_tfms=get_transforms(), size=(256,256), bs=32, num_workers=4).normalize()

print(data.classes)
print(data.c)
print(data.train_ds)
print(data.valid_ds)


print("Number of examples in training:", len(data.train_ds))
print("Number of examples in validation:", len(data.valid_ds))

xb,yb = data.one_batch()
def conv_block(ni, nf, size=3, stride=1):
    for_pad = lambda s: s if s > 2 else 3
    return nn.Sequential(
        nn.Conv2d(ni, nf, kernel_size=size, stride=stride,
                  padding=(for_pad(size) - 1)//2, bias=False), 
        nn.BatchNorm2d(nf),
        nn.LeakyReLU(negative_slope=0.1, inplace=True)  
    )

def triple_conv(ni, nf):
    return nn.Sequential(
        conv_block(ni, nf),
        conv_block(nf, ni, size=1),  
        conv_block(ni, nf)
    )

def maxpooling():
    return nn.MaxPool2d(2, stride=2)

model = nn.Sequential(
    conv_block(3, 8),
    maxpooling(),
    conv_block(8, 16),
    maxpooling(),
    triple_conv(16, 32),
    maxpooling(),
    triple_conv(32, 64),
    maxpooling(),
    triple_conv(64, 128),
    maxpooling(),
    triple_conv(128, 256),
    conv_block(256, 128, size=1),
    conv_block(128, 256),
    conv_layer(256, 3),
    Flatten(),
    nn.Linear(507, 3)
)

learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)

print(learn.summary())

learn.fit_one_cycle(1, max_lr=3e-3)

learn.export(file = Path("export.pkl"))