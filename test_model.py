# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 11:09:25 2021

@author: user
"""

import numpy as np
from PIL import Image
import math
import tensorflow as tf
import sys
import pickle
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import os
from Unet.Unet_architecture import Unet
import glob

sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
os.environ["CUDA_VISIBLE_DEVICES"]="0"

path_pggan_source = './src/pggan'
path_generator = './pretrained_models/Generator.pkl'
path_encoder = './pretrained_models/Encoder.pkl'
path_discriminator = './pretrained_models/Discriminator.h5'
path_unet = './pretrained_models/Unet.h5'

path_masks = './Dataset/masks_celebA_test/' 
path_images = './Dataset/images_celebA_test/'
path_result = './Dataset/result/'

sys.path.append(path_pggan_source)

# =============================================================================
#  load models
# =============================================================================   

def Generator(latents, is_training=False):
    TF_CONFIG = tf.ConfigProto(
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.666),
                allow_soft_placement=True)
    with tf.Session(config=TF_CONFIG) as sess: 
        with open(path_generator, 'rb') as file:
            model = pickle.load(file)
            labels = np.zeros([latents.shape[0]] + model.input_shapes[1][1:])
            images = model.run(latents, labels)
            images = images.transpose(0, 2, 3, 1)
            images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0)
            images = images[0,:,:,:]
            sess.close()
            return images 

def Encoder(img, is_training=False):
    TF_CONFIG = tf.ConfigProto(
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.666),
                allow_soft_placement=True)
    with tf.Session(config=TF_CONFIG) as sess: 
        with open(path_encoder, 'rb') as file:
            model  = pickle.load(file)
            img = img[np.newaxis,:,:,:]
            img_pre_proc = preprocess_input(img)
            z = model.predict(img_pre_proc)
            sess.close()
            return z 

def Discriminator(img, is_training=False):
    TF_CONFIG = tf.ConfigProto(
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.666),
                allow_soft_placement=True)
    with tf.Session(config=TF_CONFIG) as sess: 
        try:
            model = load_model(path_discriminator)
            img = img[np.newaxis,:,:,:]
            img_pre_proc = preprocess_input(img)
            score =  model.predict(img_pre_proc)
            return score       
        finally:
            sess.close()
              
def Unet(img, mask):
    TF_CONFIG = tf.ConfigProto(
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.666),
                allow_soft_placement=True)
    with tf.Session(config=TF_CONFIG) as sess: 
        try:
            #Load weights from pretrained model
            model_unet = Unet_arc()
            #model.summary()
            model_unet.load(
            #    './pretrained_models/different_k_models/' + str(k) + '.h5',
                path_unet,
                train_bn=True,
                lr=0.00005  
            )
            fine_img =  model_unet.predict([img[np.newaxis,:], mask[np.newaxis,:]])
            fine_img_sequeze = np.squeeze(fine_img)*255
            return fine_img_sequeze
        finally:
            sess.close()     
# =============================================================================
#  save images
# =============================================================================  

def save_images(generated_images, path, id):
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
    Image.fromarray(image.astype(np.uint8)).save(path +
                    str(id) + ".png")

# =============================================================================
#  In this part, the estimated mask content is copied to the original image to 
#  initiate another cycle.
# =============================================================================
    
def change_pixel_of_images(real_image, generated_image, mask):
    coarse_image = np.zeros(real_image.shape, dtype=np.uint8)
    coarse_image[mask == 0] = generated_image[mask == 0]
    coarse_image[mask == 255] = real_image[mask == 255]
    return coarse_image

# =============================================================================
#  In this part, a cycle is established between the encoder and the generator, 
#  and the generated image is scored at the end of each cycle using a 
#  discriminator network.
# =============================================================================

def get_coarse_image(img, mask, id_image):
    count = 0
    best_score = 0
    best_image = img
    masked_image = np.zeros(img.shape, dtype=np.uint8)
    masked_image[mask == 0] = 255
    masked_image[mask == 255] = img[mask == 255]
    generated_image = Generator(Encoder(masked_image)) 
    coarse_image = change_pixel_of_images(img, generated_image, mask)
    img_prev = np.copy(coarse_image)
    prev_score = 1000
    while(1):
        count += 1
        generated_image = Generator(Encoder(img_prev)) 
        coarse_image = change_pixel_of_images(img, generated_image, mask)
        score = Discriminator(coarse_image)
        img_prev = coarse_image
        if (score > best_score):
            best_score = score
            best_image = coarse_image
        if (score < prev_score and count > 10):
            break
        prev_score = score
    return best_image

# =============================================================================
#  In this part, the generated coarse image is taken as input and 
#  a refined result is estimated using a Unet type deep network.
# =============================================================================

def get_fine_image(img, mask, id_image):
    model_unet = Unet(img, mask)
    fine_img =  model_unet.predict([img[np.newaxis,:], mask[np.newaxis,:]])
    fine_img_sequeze = np.squeeze(fine_img)*255
    return fine_img_sequeze
    
# =============================================================================
#  test overall model
# =============================================================================  
    
for path in glob.glob(path_images + '*.png'):
    img = Image.open(path)
    id_image = (os.path.split(path)[-1]).split('.')[0]
    mask = Image.open(path_masks + str(id_image) + '.png')
    img_arr = np.array(img)
    mask_arr = np.array(mask)
    coarse_img = get_coarse_image(img_arr, mask_arr, id_image)
    course_img_arr = np.array(coarse_img)
    fine_img = get_fine_image(course_img_arr/255, mask_arr/255, id_image)
    save_images(fine_img, path_result, id_image)
    score = Discriminator(fine_img)
    print("Image : %d, Score : %.3f" % (id_image, score))
   
    with open('scores.txt', 'a') as f:
        f.write("Image: %s Score: %f\n" %(id_image, score))
    
