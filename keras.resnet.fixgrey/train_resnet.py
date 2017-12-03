#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from keras.layers import Conv2D, MaxPooling2D, Input, Activation, UpSampling2D, BatchNormalization, Add
from keras.models import Model
from dataset import get
from common import config
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint, Callback, ReduceLROnPlateau, LearningRateScheduler, EarlyStopping, TensorBoard

from keras.backend import sum, abs, sqrt, mean
import os
from keras import optimizers
import numpy as np
import sys
import argparse

parse=argparse.ArgumentParser()
parse.add_argument('-c', '--premodel', type=str, default = None)
parse.add_argument('-l','--lr', type=float, default = 1e-3)
args=parse.parse_args()

base_model = ResNet50(weights = 'imagenet', include_top = False, input_tensor = Input(shape=(None, None, config.nr_channel)))

addlayers = []
for layer in base_model.layers:
    if 'add' in layer.name:
        addlayers.append(layer)
inds = [15, 12, 6]
pyramids = [addlayers[ind].output for ind in inds]

x = pyramids[0]
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(config.nr_heatmap_channel, (1, 1), activation='linear', padding='same')(x)
base_output = x
for x in pyramids[1:]:
    base_output = UpSampling2D(size = (2, 2))(base_output)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(config.nr_heatmap_channel, (1, 1), activation='linear', padding='same')(x)
    base_output = Add()([base_output, x])

x = base_output
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(config.nr_heatmap_channel, (1, 1), activation='relu', padding='same')(x)
pred_segmap = UpSampling2D(size = (8, 8))(x)
model = Model(inputs=base_model.input, outputs=pred_segmap)

curdir = os.path.abspath('.')
curname = os.path.split(curdir)[-1]
logname = os.path.join(os.path.split(curdir)[0], 'logs')
newlogdir = os.path.join(logname, curname)
if os.path.exists(newlogdir) is False:
    os.mkdir(newlogdir)
    os.mkdir(os.path.join(newlogdir, 'models'))
    os.system('ln -s '+newlogdir+' ./logs')

def mae(segmap, pred_segmap):
    pred_cnt = sum(pred_segmap, axis=(1,2,3)) / 20
    cnt = sum(segmap, axis=(1,2,3)) / 20
    mae = mean(abs(pred_cnt - cnt))
    return mae

def mse(segmap, pred_segmap):
    pred_cnt = sum(pred_segmap, axis=(1,2,3)) / 20
    cnt = sum(segmap, axis=(1,2,3)) / 20
    mse = sqrt(mean((pred_cnt - cnt)**2))
    return mse

def true_num(segmap, pred_segmap):
    true_nums = sum(segmap, axis=(1,2,3)) / 20
    true_num = mean(true_nums)
    return true_num

def pred_num(segmap, pred_segmap):
    pred_nums = sum(pred_segmap, axis=(1,2,3)) / 20
    pred_num = mean(pred_nums)
    return pred_num

if args.premodel:
    model.load_weights(args.premodel)
#train
adam = optimizers.Adam(lr = args.lr)
model.compile(optimizer = adam, loss='mean_squared_error', metrics=[true_num, pred_num, mae, mse])

# serialize model to JSON
model_json = model.to_json()
with open("./logs/model.json", "w") as json_file:
    json_file.write(model_json)
          
checkpoint = ModelCheckpoint('./logs/models/weights_{epoch:02d}.hdf5', verbose=1, save_best_only=False, save_weights_only=True, mode='auto', period = 1)

data_generator = get('train')
model.fit_generator(data_generator, steps_per_epoch = config.per_epoch, epochs=config.nr_epoch, callbacks=[checkpoint])
