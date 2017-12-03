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
            open(os.path.join(base_dir, 'ShanghaiTech_Crowd-mat.pkl'), 'rb'))
    
    def ensure_size(rng, img, hms, shape):
        ch, cw = min(img.shape[0], shape[0]), min(img.shape[1], shape[1])
        #random crop
        dx, dy = img.shape[1] - cw, img.shape[0] - ch
        x0, y0 = rng.randint(max(0, dx + 1)), rng.randint(max(0, dy + 1))
        x1, y1 = x0 + cw, y0 + ch
        img = img[y0:y1, x0:x1]
        hms = [hm[y0:y1, x0:x1] for hm in hms]
        #pad to target shape
        img, pad = pad_image_to_shape(img, shape, return_padding=True)
        hms = [pad_image_to_shape(hm, shape) for hm in hms]
        return img, hms, (x0 - pad[1], y0 - pad[0])

    def data_gen(dataset_name, items):
        if dataset_name == 'train':
            items = [x for x in items if 'train' in x['group']]
        else:
            items = [x for x in items if 'test' in x['group']]
        while True:
            if dataset_name == 'train':
                np.random.shuffle(items)
            imgs = []
            segmaps = []
            for item in items:
                try:
                    img = cv2.imdecode(np.fromstring(item['nr_data'], np.uint8), cv2.IMREAD_UNCHANGED)
                    if img.ndim == 2:#convert grey to bgr
                        img = img.reshape(img.shape + (1,))
                        img = np.tile(img, (1, 1, 3))
                    else:
                        img = img[:, :, :3]
                    segmap = [np.load(BytesIO(item['mat_nori_data']))]
                except Exception as e:
                    import IPython
                    IPython.embed()
                    #print(e)
                    continue
                img, segmap, _ = ensure_size(
                        rng, img, segmap, config.image_shape)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgs.append(img)
                segmaps.append(np.transpose(np.array(segmap), (1, 2, 0)))
                if len(imgs) == config.minibatch_size:
                    imgs = preprocess_input(np.array(imgs).astype(np.float64))
                    segmaps = np.array(segmaps)
                    yield (imgs, segmaps)
                    imgs = []
                    segmaps = []

    return data_gen(dataset_name, items)
