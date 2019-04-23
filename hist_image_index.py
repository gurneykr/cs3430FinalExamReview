#!/usr/bin/python

import argparse
import cv2
import sys
import os
import re
## use import pickle in Py3.
import pickle

################################
# module: hist_image_index.py
# Krista Gurney
# A01671888
################################
from os import listdir
from os.path import isfile, join

## indexing dictionary.
HIST_INDEX = {}

def hist_index_img(imgp, color_space, bin_size=8):
    image = cv2.imread(imgp)
    if color_space == 'rgb':
        input_hist = cv2.calcHist([image], [0, 1, 2], None, [bin_size, bin_size, bin_size], [0, 256, 0, 256, 0, 256])
        norm_hist = cv2.normalize(input_hist, input_hist).flatten()
    elif color_space == 'hsv':
        input_hist = cv2.calcHist([image], [0, 1, 2], None, [bin_size, bin_size, bin_size], [0, 180, 0, 180, 0, 180])
        norm_hist = cv2.normalize(input_hist, input_hist).flatten()
    else:
        raise Exception("Unknown color space")

    return imgp, norm_hist

def hist_index_img_dir(imgdir, color_space, bin_size, pick_file):
    print(imgdir)
    os.listdir(imgdir)
    onlyfiles = [f for f in listdir(imgdir) if isfile(join(imgdir, f))]
    for filename in onlyfiles:
        filepath = imgdir + "\\"+ filename
        imgdr, norm_hist = hist_index_img(filepath, color_space, bin_size)
        HIST_INDEX[imgdr] = norm_hist

    outfile = open(pick_file, 'wb')
    pickle.dump(HIST_INDEX, outfile)
    outfile.close()

    # print(HIST_INDEX)
    print('indexing finished')


## ========================= Image Indexing Tests =====================
  
## change these as you see fit.
## IMGDIR is the directory where the images to be indexed are saved.
## PICDIR is the directory where pickled dictionaries are saved.
# IMGDIR = 'C:\\Users\\Krista Gurney\\Documents\\cs3430\\hw12Starter\\images'
IMGDIR = 'images'
PICDIR = 'picks\\'

# PICDIR = 'C:\\Users\\Krista Gurney\\Documents\\cs3430\\hw12Starter\\picks\\'
# PICDIR = '/home/vladimir/teaching/CS3430/S19/hw/hw12f/hist_indexing/picks/'

def test_01():
    HIST_INDEX = {}
    hist_index_img_dir(IMGDIR, 'rgb', 8, PICDIR + 'rgb_hist8.pck')

def test_02(): 
    HIST_INDEX = {}
    hist_index_img_dir(IMGDIR, 'rgb', 16, PICDIR + 'rgb_hist16.pck')

def test_03():
    HIST_INDEX = {}
    hist_index_img_dir(IMGDIR, 'hsv', 8, PICDIR + 'hsv_hist8.pck')

def test_04():
    HIST_INDEX = {}
    hist_index_img_dir(IMGDIR, 'hsv', 16, PICDIR + 'hsv_hist16.pck')


if __name__ == '__main__':
    test_01()


