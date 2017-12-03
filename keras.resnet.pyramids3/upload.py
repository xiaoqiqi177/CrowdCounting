#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tornado.ioloop
import tornado.web
import cv2
import glob
import pickle
import numpy as np
import argparse
import os
import math
from utils import padimg
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model, model_from_json
from io import BytesIO
from PIL import Image

json_file = open('./logs/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
mdl_pth = './logs/models/weights_90.hdf5'
model.load_weights(mdl_pth)

def predict(img):
    padded_img = padimg(img, 32)
    img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
    test_data = preprocess_input(np.array([img]).astype(np.float64))
    pred_heatmap = model.predict(test_data, batch_size=1)[0]
    k = 255/pred_heatmap.max()
    vis_heatmap = (pred_heatmap * k).astype('uint8')
    vis_heatmap = np.concatenate((vis_heatmap, vis_heatmap, vis_heatmap), axis=2)
    retimg = np.concatenate((padded_img, vis_heatmap), axis=0)
    pred_num = pred_heatmap.sum() / 20
    return k, pred_num, retimg

class UploadFileHandler(tornado.web.RequestHandler):
    def get(self):
        self.write('''
        <html>
          <head><title>Upload File</title></head>
          <body>
            <form action='file' enctype="multipart/form-data" method='post'>
            <br/>
            <input type='file' name='file1'/><br/>
            <br/>
            <input type='submit' value='submit'/>
            </form>
          </body>
        </html>
        ''')

    def post(self):
        upload_path=os.path.join(os.path.dirname(__file__),'files')  
        
        file_metas=self.request.files['file1']    
        for meta in file_metas:
            filename=meta['filename']
            filepath1=os.path.join(upload_path,filename)
            with open(filepath1,'wb') as up:      
                up.write(meta['body'])
        img = cv2.imread(filepath1)
        k, pred_num, retimg = predict(img)
        print(k)
        print(pred_num)
        #self.write(str(pred_num))
        retimg = retimg[:,:, ::-1]
        showimg = Image.fromarray(retimg)
        im_file = BytesIO()
        showimg.save(im_file, format='png')
        im_data = im_file.getvalue() 
        self.set_header('Content-type', 'image/png')
        self.set_header('Content-length', len(im_data))   
        self.write(im_data)

def make_app():
    app=tornado.web.Application([
        (r"/file",UploadFileHandler),
    ])
    return app

if __name__ == "__main__":
    app = make_app()
    app.listen(9001)
    tornado.ioloop.IOLoop.instance().start()

