#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import cv2
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
import numpy as np
from common import config
from utils import padimg
oriimg = cv2.imread(sys.argv[1])
padded_img = padimg(oriimg, 32)
img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
test_data = preprocess_input(np.array([img]).astype(np.float64))
mdl_pth = './logs/models/weights_01.hdf5'
model = load_model(mdl_pth)
pred_heatmap = model.predict(test_data, batch_size=1)[0]
b = np.zeros(pred_heatmap.shape)
g = np.clip(pred_heatmap*2, 0, 255)
r = np.clip(255-pred_heatmap*2, 0, 255)


vis_heatmap = np.concatenate((b, g, r), axis=2)
#cv2.imshow('img', np.concatenate((padded_img, vis_heatmap), axis=1))
pred_num = pred_heatmap.sum() / 20
print('predicted num is ', pred_num)
