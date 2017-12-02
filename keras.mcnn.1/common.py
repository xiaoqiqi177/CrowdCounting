#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class Config:
    minibatch_size = 12
    nr_channel = 3  # input channels
    image_shape = (512, 512)
    nr_heatmap_channel = 1
    input_shape = image_shape + (nr_channel, )
    segmap_shape = image_shape + (nr_heatmap_channel, )
    nr_epoch = 100
    per_epoch = 200
config = Config()
