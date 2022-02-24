# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 09:11:00 2019

@author: user
"""

import tensorflow as tf
import sys
import pickle
import numpy as np
from PIL import Image
import math
path_pg_gan_code = './'
path_model = './model/128_final.pkl'
path_images='./FID_test'


sys.path.append(path_pg_gan_code)

def combine_and_save_images(generated_images, e):
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
    image = np.clip(np.rint((image + 1.0) / 2.0 * 255.0), 0.0, 255.0)
    Image.fromarray(image.astype(np.uint8)).save(path_images+'/'+
                    str(e)+"_"+".png")   
    
    
def Generator( z_vector, is_training=False):
  with tf.Session() as sess:

    # Import official CelebA-HQ networks.
    try:
        with open(path_model, 'rb') as file:
            G, D, Gs = pickle.load(file)
            z=z_vector.eval()
            latents = z
            # Generate dummy labels (not used by the official networks).
            labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])
            # Run the generator to produce a set of images.
            images = Gs.run(latents, labels)
            images = images.transpose(0, 2, 3, 1)
            
            return images
            
                
    except FileNotFoundError:
        print('before running the code, download pre-trained model to project_root/asset_model/')
        raise
k=19713       
for i in range(20000):
        
    random_z = tf.random.normal(shape=[256, 128])
    img=Generator(random_z)
    shape=np.shape(img)
#    print(shape)
#    print(shape[0])
    for j in range(shape[0]):
        images=img[j]
        images=images[np.newaxis,:,:,:]
#        print(np.shape(images))
        combine_and_save_images(images,k)
        k+=1
        print(k)