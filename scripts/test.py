#!/usr/bin/env mdl
# -*- coding: utf-8 -*-

import os
import cv2
import pickle
import argparse
import numpy as np
import nori2 as nori
import logging
from io import BytesIO
from nori2.multi import MultiSourceReader
from tqdm import tqdm
from tabulate import tabulate

from meghair.utils.imgproc import imdecode
from meghair.utils.io import load_network
from megskull.graph import Function
from meghair.utils import logconf
from neupeak.utils import imgproc
from neupeak.utils.misc import i01c_to_ic01, list2nparray

logconf.set_default_level(logging.WARNING)
logconf.set_mgb_log_level(logging.WARNING)

base_dir = '/unsullied/sharefs/_research_video/VideoData/datasets/Public/Counting/ShanghaiTech_Crowd_Counting_Dataset/'
items = pickle.load(
    open(os.path.join(base_dir, 'ShanghaiTech_Crowd-mat.pkl'), 'rb'))
items_a = [x for x in items if 'A-test' in x['group']]
items_b = [x for x in items if 'B-test' in x['group']]
nr = MultiSourceReader(patterns=[os.path.join(base_dir, '*.nori')])


def worker(model_path):
    net = load_network(model_path)
    pred_func_nchw = Function().compile(net.outputs)

    def do_work(items):
        cnt_pred, cnt_gt = [], []
        for item in tqdm(items):
            img = imdecode(nr.get(item['nori_id']))
            img = imgproc.pad_image_size_to_multiples_of(
                img, 32, align='top-left')
            data = np.ascontiguousarray(i01c_to_ic01(
                img)[np.newaxis]).astype(np.float32)
            out = pred_func_nchw(data)
            hmap, cnt = out[0][0][0], out[1][0]
            cnt_pred.append(cnt)
            cnt_gt.append(item['cnt'])

        cnt_pred, cnt_gt = np.array(cnt_pred), np.array(cnt_gt)
        diff = cnt_pred - cnt_gt
        mae = np.abs(diff).mean()
        mse = np.sqrt((diff**2).mean())
        return mae, mse

    part_a = do_work(items_a)
    part_b = do_work(items_b)
    return part_a, part_b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='network')
    args = parser.parse_args()

    if os.path.isdir(args.network):
        model_path = [os.path.join(args.network, x)
                      for x in os.listdir(args.network) if 'epoch' in x]
        model_path = sorted(model_path, key=lambda x: int(x.split('_')[-1]))
    else:
        model_path = [args.network]

    table = [["model", "MAE-a", "MSE-a", "MAE-b", "MSE-b"]]
    for m in tqdm(model_path):
        part_a, part_b = worker(m)
        table.append([os.path.basename(m), *part_a, *part_b])

    print(args.network)
    print('==============================')
    print(tabulate(table, floatfmt=".3f"))

if __name__ == '__main__':
    main()
