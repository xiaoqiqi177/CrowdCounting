#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from common import config
import cv2
import random
import glob
import pickle
from io import BytesIO
import os
from utils import stable_rng, pad_image_to_shape
from keras.applications.imagenet_utils import preprocess_input

def get(dataset_name):
    base_dir = '/unsullied/sharefs/xqq/temp/ShanghaiTech_Crowd/'
    rng = stable_rng(stable_rng)
    items = pickle.load(
            open(os.path.join(base_dir, 'ShanghaiTech_Crowd-mat-uniform.pkl'), 'rb'))
    
    def data_gen(dataset_name, items):
        if dataset_name == 'train':
            items = [x for x in items if 'train' in x['group']]
        else:
            items = [x for x in items if 'test' in x['group']]
        imgs = []
        segmaps = []
        for item in items:
            img = cv2.imdecode(np.fromstring(item['nr_data'], np.uint8), cv2.IMREAD_UNCHANGED)
            if img.ndim == 2:#convert grey to bgr
                img = img.reshape(img.shape + (1,))
                img = np.tile(img, (1, 1, 3))
            else:
                img = img[:, :, :3]
            segmap = [np.load(BytesIO(item['mat_nori_data']))]
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)
            segmaps.append(np.transpose(np.array(segmap), (1, 2, 0)))
        #imgs = preprocess_input(np.array(imgs).astype(np.float64))
        segmaps = np.array(segmaps)
        return (imgs, segmaps)

    return data_gen(dataset_name, items)
