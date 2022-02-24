# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 22:40:39 2020

@author: user
"""

from PIL import Image, ImageFilter, ImageEnhance
from random import randrange, uniform
import os
import pathlib
from util import MaskGenerator
import math
import numpy as np
import copy

path_main_folder = './dataset/real/'
path_high_artifacts_images = './dataset/high_artifacts/'
path_very_small_artifacts_images = './dataset/very_small_artifacts/'
path_mask = './dataset/masks/'
mask_generator = MaskGenerator(128, 128, 3)

def save_images(generated_images, path, image_name):
    generated_images=generated_images[np.newaxis,:,:,:]
    num = generated_images.shape[0]  
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:]
    
    image = np.zeros((height*shape[0], width*shape[1],shape[2]),
                     dtype=generated_images.dtype)
    
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1],:] = \
        img[:, :, :]
#    image = np.clip(np.rint((image + 1.0) / 2.0 * 255.0), 0.0, 255.0)
    Image.fromarray(image.astype(np.uint8)).save(path + image_name)
    
    
# =============================================================================
#  For create fake images applied high artifacts
# =============================================================================

pathlib.Path(path_high_artifacts_images).mkdir(parents=True, exist_ok=True) 

for image_name in os.listdir(path_main_folder):
   # print(image_name)
   mask = mask_generator.sample()
   save_images(mask*255, path_mask, image_name)
   path_image = os.path.join(path_main_folder, image_name)
   image = Image.open(path_image)
   crop_img = copy.copy(image) 
   
   random_value_for_processing = randrange(0,3) #0-1-2
   
   if random_value_for_processing == 0: #GaussianBlur
       random_radius = uniform(1, 2.5)
       image_with_artifacts = crop_img.filter(ImageFilter.GaussianBlur(radius=random_radius))
   elif random_value_for_processing == 1: #Contrast
       scale_value = uniform(0.4, 0.8)
       image_with_artifacts = ImageEnhance.Contrast(crop_img).enhance(scale_value)
   else: #Brightness
       scale_value = uniform(0.4, 0.8)
       image_with_artifacts = ImageEnhance.Brightness(crop_img).enhance(scale_value)
             
   img_save = (1-mask) * image_with_artifacts + mask * image     
   save_images(img_save, path_high_artifacts_images, image_name)
   
# =============================================================================
#  For create fake images applied very little artifacts
# =============================================================================
   
# pathlib.Path(path_very_small_artifacts_images).mkdir(parents=True, exist_ok=True) 

# for i in range(8000):
#   random_image_id = randrange(0,29000) 
#   mask = mask_generator.sample()
#   path_image = path_main_folder + str(random_image_id) + '.jpg'
#   image = Image.open(path_image)
#   save_images(mask*255, path_mask, str(random_image_id) + '.jpg')
# #   box = (32, 32, 96, 96)
# #   crop_img = image.crop(box)
#   crop_img = copy.copy(image) 
#   random_value_for_processing = randrange(0,3) #0-1-2
#   if random_value_for_processing == 0: #GaussianBlur
#       random_radius = uniform(0.3, 0.8)
#       image_with_artifacts = crop_img.filter(ImageFilter.GaussianBlur(radius=random_radius))
#   elif random_value_for_processing == 1: #Contrast
#       scale_value = uniform(0.85, 0.95)
#       image_with_artifacts = ImageEnhance.Contrast(crop_img).enhance(scale_value)
#   else: #Brightness
#       scale_value = uniform(0.85, 0.95)
#       image_with_artifacts = ImageEnhance.Brightness(crop_img).enhance(scale_value)
  
#   img_save = (1-mask) * image_with_artifacts + mask * image     
#   save_images(img_save, path_very_small_artifacts_images, str(random_image_id) + '.jpg')   
