# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 10:02:15 2021

@author: user
"""

#for dlib: 1-pip install cmake 2- conda install -c conda-forge dlib
import numpy as np
from PIL import Image
import dlib
from skimage.draw import polygon
import random
#print(dlib.__version__)

path_real_image = './Dataset/ffhq_128/' 
path_save_mask = './Dataset/masks_ffhq_test/'
path_real_save_image = './Dataset/images_ffhq_test/'
    
def get_mask(path, id_img):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./pretrained_models/shape_predictor_68_face_landmarks.dat')
    img = dlib.load_rgb_image(path)
    rect = detector(img)[0]
    sp = predictor(img, rect)
    landmarks = np.array([[p.x, p.y] for p in sp.parts()])
    outline = landmarks[[*range(17), *range(26,16,-1)]]
    Y, X = polygon(outline[:,1], outline[:,0])
    mask = np.zeros(img.shape, dtype=np.uint8)
    mask[Y, X] = 1#img[Y, X]
    mask = 1 - mask
    mask *= 255
    Image.fromarray(mask).save(path_save_mask + str(id_img) + '.png') 
    return (mask/255)

for i in range(1):
    try:
        # print(i)
        id_random = random.randint(0, 69999)
        path = path_real_image + str(id_random) + '.png'
        get_mask(path, id_random)
        img = Image.open(path)
        img.save(path_real_save_image + str(id_random) +'.png')
    except:
        print("An exception occurred")
        