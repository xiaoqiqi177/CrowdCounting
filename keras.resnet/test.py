#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import cv2
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model, model_from_json
import numpy as np
from utils import padimg
import argparse

parse=argparse.ArgumentParser()
parse.add_argument('-m', '--modelpath', type=str)
parse.add_argument('-i', '--imgpath', type=str)
args=parse.parse_args()

oriimg = cv2.imread(args.imgpath)
padded_img = padimg(oriimg, 32)
img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
test_data = preprocess_input(np.array([img]).astype(np.float64))

json_file = open('./logs/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
#loaded weights

mdl_pth = args.modelpath
model.load_weights(mdl_pth)

pred_heatmap = model.predict(test_data, batch_size=1)[0]
b = np.zeros(pred_heatmap.shape)
g = np.clip(pred_heatmap*2, 0, 255)
r = np.clip(255-pred_heatmap*2, 0, 255)

vis_heatmap = np.concatenate((b, g, r), axis=2)
cv2.imshow('img', np.concatenate((padded_img, vis_heatmap), axis=1))
pred_num = pred_heatmap.sum() / 20
print('predicted num is ', pred_num)
