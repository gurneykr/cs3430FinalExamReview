#!/usr/bin/python

import argparse
import cv2
import sys
import os
## use import pickle in Py3
import pickle
from matplotlib import pyplot as plt
from os.path import basename
from hist_image_index import hist_index_img

################################
# module: hist_image_index.py
# Krista Gurney
# A01671888
################################

#
# def compute_hist_sim(inhist_vec, hist_index, hist_sim, topn=3):
#   ## your code here
#   pass
 
def show_images(input_image, match_list):

  # show original image
  rgb1 = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
  fig1 = plt.figure(1)
  fig1.suptitle('Input Image')
  plt.imshow(rgb1)
  plt.show()

  for path, sim_score in match_list:
    image = cv2.imread(path)
    rgb1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig1 = plt.figure(1)
    fig1.suptitle(path + ' Sim = '+str(sim_score))
    plt.imshow(rgb1)
    plt.show()


def find_sim_rgb_images(imgpath, bin_size, hist_index, hist_sim):
  return find_sim_images(imgpath, bin_size, hist_index, hist_sim, 'rgb')

def find_sim_hsv_images(imgpath, bin_size, hist_index, hist_sim):
  return find_sim_images(imgpath, bin_size, hist_index, hist_sim, 'hsv')

def find_sim_images(imgpath, bin_size, hist_index, hist_sim, image_type):
  _path, norm_hist1 = hist_index_img(imgpath, image_type, bin_size)

  sim_arr = []
  reverse_flag = True

  for path,norm_hist2 in hist_index.items():
    if hist_sim == 'inter':
      sim_score = cv2.compareHist(norm_hist1, norm_hist2, cv2.HISTCMP_INTERSECT)
    elif hist_sim == 'chisqr': #closer to 0 more similar
      sim_score = cv2.compareHist(norm_hist1, norm_hist2, cv2.HISTCMP_CHISQR)
      reverse_flag = False
    elif hist_sim == 'bhatta':#closer to 0 more similar
      sim_score = cv2.compareHist(norm_hist1, norm_hist2, cv2.HISTCMP_BHATTACHARYYA)
      reverse_flag = False
    elif hist_sim == 'correl':
      sim_score = cv2.compareHist(norm_hist1, norm_hist2, cv2.HISTCMP_CORREL)

    sim_arr.append((path, sim_score))

  sorted_sim = sorted(sim_arr, key=lambda tup: tup[1], reverse = reverse_flag)
  return sorted_sim[0:3]


def load_hist_index(pick_path):
  with open(pick_path, 'rb') as histfile:
    return pickle.load(histfile)

## change the paths as you see fit and add calls to show_images()
## after you implement it.


## ============= Image Retrieval Tests ===========================

## change these as you see fit.
## IMGDIR is the directory for test images
## PICDIR is the directory where the pickle files are stored.

IMGDIR = 'C:\\Users\\Krista Gurney\\Documents\\cs3430\\hw12Starter\\hist_test\\'
PICDIR = 'C:\\Users\\Krista Gurney\\Documents\\cs3430\\hw12Starter\\picks\\'
'''
My Py shell output:
images/123461762.JPG --> 2.69072864504
images/123465049.JPG --> 2.63319342056
images/123472255.JPG --> 2.43531483644
'''
def test_01():
  hist_index = load_hist_index(PICDIR + 'rgb_hist8.pck')
  assert len(hist_index) == 318
  imgpath = IMGDIR + 'food_test\\img01.JPG'
  inimg = cv2.imread(imgpath)
  top_matches = find_sim_rgb_images(imgpath, 8, hist_index, 'inter')

  # rgb1 = cv2.cvtColor(inimg, cv2.COLOR_BGR2RGB)
  # fig1 = plt.figure(1)
  # fig1.suptitle('Input Image')
  # plt.imshow(rgb1)

  for imagepath, sim in top_matches:
    print(imagepath + ' --> ' + str(sim))
  show_images(inimg, top_matches)
  del hist_index

'''
My Py shell output:
>>> test_02()
images/123472992.JPG --> 1.04123155377
images/123465793.JPG --> 0.778828541127
images/123465726.JPG --> 0.775194820913
'''
def test_02():
  hist_index = load_hist_index(PICDIR + 'hsv_hist8.pck')
  assert len(hist_index) == 318
  imgpath = IMGDIR + 'food_test/img03.JPG'
  inimg = cv2.imread(imgpath)
  top_matches = find_sim_hsv_images(imgpath,
		                    8, hist_index, 'inter')
  for imagepath, sim in top_matches:
    print(imagepath + ' --> ' + str(sim))
  show_images(inimg, top_matches)
  del hist_index

'''
My Py shell output:
images/123465245.JPG --> 15.8357133494
images/17_02_21_22_17_56_orig.png --> 21.0158345761
images/17_02_21_22_17_55_orig.png --> 21.401725557
'''
def test_03():
  hist_index = load_hist_index(PICDIR + 'rgb_hist8.pck')
  assert len(hist_index) == 318
  imgpath = IMGDIR + 'food_test/img04.JPG'
  inimg = cv2.imread(imgpath)
  top_matches = find_sim_rgb_images(imgpath,
		                    8, hist_index, 'chisqr')
  for imagepath, sim in top_matches:
    print(imagepath + ' --> ' + str(sim))
  show_images(inimg, top_matches)
  del hist_index

'''
My Py shell output:
images/17_02_21_22_14_24_orig.png --> 0.0952925097908
images/17_02_21_22_14_14_orig.png --> 0.190314746298
images/17_02_21_22_20_56_orig.png --> 0.282203709903
'''
def test_04():
  hist_index = load_hist_index(PICDIR + 'rgb_hist16.pck')
  assert len(hist_index) == 318
  imgpath = IMGDIR + 'car_test/img22.png'
  inimg = cv2.imread(imgpath)
  top_matches = find_sim_rgb_images(imgpath,
		                    16, hist_index, 'bhatta')
  for imagepath, sim in top_matches:
    print(imagepath + ' --> ' + str(sim))
  show_images(inimg, top_matches)
  del hist_index

'''
My Py shell output:
images/123472992.JPG --> 0.948968044156
images/123459060.JPG --> 0.957500781094
images/123465726.JPG --> 0.957573532491

Matching car_test/img023.png in HSV space on bhatta doesn't produce
good results.
''' 
def test_05():
  hist_index = load_hist_index(PICDIR + 'hsv_hist16.pck')
  assert len(hist_index) == 318
  imgpath = IMGDIR + 'car_test/img02.png'
  inimg = cv2.imread(imgpath)
  top_matches = find_sim_hsv_images(imgpath,
		                    16, hist_index, 'bhatta')
  for imagepath, sim in top_matches:
    print(imagepath + ' --> ' + str(sim))
  show_images(inimg, top_matches)
  del hist_index

'''
My Py shell output:
images/16_07_02_14_21_01_orig.png --> 0.0601641627891
images/16_07_02_14_21_06_orig.png --> 0.0626254148808
images/16_07_02_14_21_02_orig.png --> 0.0641319684534

Matching car_test/img023.png in RGB space on bhatta produces
excellent matches.
'''
def test_06():
  hist_index = load_hist_index(PICDIR + 'rgb_hist16.pck')
  assert len(hist_index) == 318
  imgpath = IMGDIR + 'car_test/img02.png'
  inimg = cv2.imread(imgpath)
  top_matches = find_sim_rgb_images(imgpath,
		                    16, hist_index, 'bhatta')
  for imagepath, sim in top_matches:
    print(imagepath + ' --> ' + str(sim))
  show_images(inimg, top_matches)
  del hist_index


  
 
if __name__ == '__main__':
  # hist_index = load_hist_index(PICDIR +'rgb_hist8.pck')
  # print(len(hist_index))
  test_01()

