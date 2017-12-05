#!/usr/bin/env mdl

import cv2
import pickle
import numpy as np
import multiprocessing
from io import BytesIO

from scipy.spatial import distance
from scipy.ndimage import gaussian_filter

from tqdm import tqdm
from IPython import embed

def process(data):
    img, d = data
    shape, pos = img.shape, d['pos']
    h, w = shape[:2]
    mat = np.zeros(shape[:2])
    for i in range(len(pos)):
        m = np.zeros(shape[:2])
        x, y = int(pos[i][0]), int(pos[i][1])
        if y < h and x < w:
            m[y, x] = 20
            mat += gaussian_filter(m, 8)
    return mat, img, d


def main():
    data = pickle.load(open(
        '/unsullied/sharefs/xqq/temp/ShanghaiTech_Crowd/ShanghaiTech_Crowd.pkl', 'rb'))

    def data_iter():
        for d in data:
            img = cv2.imdecode(np.fromstring(d['nr_data'], np.uint8), cv2.IMREAD_UNCHANGED)
            assert img is not None, 'failed to decode'
            yield (img, d)

    pool = multiprocessing.Pool(8)
    result = []
    for m in tqdm(pool.imap(process, data_iter())):
        mat, img, d = m
        bio = BytesIO()
        np.save(bio, mat)
        mat_nr_data = bio.getvalue()
        d.update({'mat_nori_data': mat_nr_data})
        result.append(d)
        if False:
            cv2.imshow("img", img)
            cv2.imshow("mat", mat)
            cv2.waitKey(0)

    pickle.dump(result, open(
        '/unsullied/sharefs/xqq/temp/ShanghaiTech_Crowd/ShanghaiTech_Crowd-mat-uniform.pkl', 'wb'))

if __name__ == '__main__':
    main()
