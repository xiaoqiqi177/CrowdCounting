#!/usr/bin/env python3

import os
import cv2
import scipy.io as sio
import pickle
from tqdm import tqdm
from IPython import embed

D = [['A-train', '/unsullied/sharefs/xqq/temp/ShanghaiTech_Crowd/part_A_final/train_data'],
     ['A-test', '/unsullied/sharefs/xqq/temp/ShanghaiTech_Crowd/part_A_final/test_data'],
     ['B-train', '/unsullied/sharefs/xqq/temp/ShanghaiTech_Crowd/part_B_final/train_data'],
     ['B-test', '/unsullied/sharefs/xqq/temp/ShanghaiTech_Crowd/part_B_final/test_data']]

data = []

for n, d in D:
    img_path = os.path.join(d, 'images')
    gt_path = os.path.join(d, 'ground_truth')
    for f in tqdm(os.listdir(img_path)):
        with open(os.path.join(img_path, f), 'rb') as fp:
            nr_data = fp.read()
        mat = sio.loadmat(os.path.join(
            gt_path, 'GT_' + f.replace('.jpg', '.mat')))
        mat = mat['image_info'][0][0][0][0]
        data.append({
            'group': n,
            'nr_data': nr_data,
            'pos': mat[0],
            'cnt': mat[1][0][0],
        })

pickle.dump(data, open(
    '/unsullied/sharefs/xqq/temp/ShanghaiTech_Crowd/ShanghaiTech_Crowd.pkl', 'wb'))
