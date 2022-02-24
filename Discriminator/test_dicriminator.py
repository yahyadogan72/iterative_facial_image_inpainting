# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 14:19:41 2021

@author: user
"""

import glob
import os
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model

path_model = './pretrained_models/'
model = load_model(path_model+'discriminator.pkl')
path_images = '...'

new_width  = 128
new_height = 128

for path in glob.glob(path_images + '*.png'):
    image_id = (os.path.split(path)[-1]).split('.')[0]
    img_org = image.load_img(path)   
    img = img_org.resize((new_width, new_height))        
    img = image.img_to_array(img)
    img = preprocess_input(img)
    img = img[np.newaxis,:,:,:]
    scor = model.predict(img)
    print("ID : %d, Score : %.3f" % (image_id, scor))
