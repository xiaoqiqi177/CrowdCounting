#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import cv2
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model, model_from_json
import numpy as np
from utils import padimg
import argparse
from dataset_test import get
from keras.applications.imagenet_utils import preprocess_input

parse=argparse.ArgumentParser()
parse.add_argument('-m', '--modelpath', type=str)
args=parse.parse_args()

json_file = open('./logs/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
#loaded weights

mdl_pth = args.modelpath
model.load_weights(mdl_pth)

imgs, segmaps = get('test')
datanum = len(imgs)
mae = 0
mse = 0 
for img, segmap in zip(imgs, segmaps):
    padded_img = padimg(img, 32)
    test_data = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
    test_data = preprocess_input(np.array([test_data]).astype(np.float64))
    pred_heatmap = model.predict(test_data, batch_size=1)[0]
    k = 255/pred_heatmap.max()
    vis_heatmap = (pred_heatmap * k).astype('uint8')
    vis_heatmap = np.concatenate((vis_heatmap, vis_heatmap, vis_heatmap), axis=2)
    retimg = np.concatenate((padded_img, vis_heatmap), axis=0)
    pred_num = pred_heatmap.sum() / 20
    num = segmap.sum() / 20
    print('pred / num: ', pred_num, '/', num)
    #cv2.imshow('retimg', retimg)
    #cv2.waitKey(0)
    mae += np.abs(pred_num - num)
    mse += (pred_num - num)**2
print('mae: ', mae/datanum)
print('mse: ', np.sqrt(mse/datanum))
