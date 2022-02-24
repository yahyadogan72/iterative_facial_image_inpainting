# -*- coding: utf-8 -*-
"""
Created on Sat May  9 16:26:16 2020

@author: user
"""

import os
import datetime
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback
import matplotlib.pyplot as plt
from Unet_architecture import Unet

BATCH_SIZE = 16
# =============================================================================
#  Data loader
# =============================================================================
class AugmentingDataGenerator(ImageDataGenerator):
    def flow_from_directory(self, train_val_test, *args, **kwargs):
        
        directory =  './dataset/' + train_val_test
        gen = super().flow_from_directory(directory, class_mode=None, *args, **kwargs)     
        path_mask = './dataset/' + train_val_test + '/mask' 
        path_output = './dataset/' + train_val_test + '/output'
        while True:
            # Get augmentend image samples
            input_images = []
            masks = []
            output_images = []
            
            index = next(gen.index_generator)

            for i in range(BATCH_SIZE):
                image_name = gen.filenames[index[i]]
                head, tail = os.path.split(image_name)
                
                filename1 = './dataset/' + train_val_test + '/input'+ '/input/' + tail
                filename2 = './dataset/' + train_val_test + '/mask/' + tail
                filename3 = './dataset/' + train_val_test + '/output/' + tail
                
                if os.access(filename1, os.W_OK) and os.access(filename2, os.W_OK) and os.access(filename3, os.W_OK):
                    image_input = plt.imread(directory + '/input/' + tail)
                    image_mask = plt.imread(path_mask + '/' + tail)
                    image_output = plt.imread(path_output + '/' + tail)
                    
                    input_images.append(image_input)
                    masks.append(image_mask)
                    output_images.append(image_output)
                else:
                    print(image_name)
                    index = next(gen.index_generator)
                    i = 0
                    input_images.clear
                    masks.clear
                    output_images.clear
            yield [np.array(input_images), np.array(masks)], np.array(output_images)

# Create training generator
train_datagen = AugmentingDataGenerator(  
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rescale=1./255,
    horizontal_flip=True
)

# Create training generator
train_generator = train_datagen.flow_from_directory(
    'train', 
    target_size=(128, 128), 
    batch_size=BATCH_SIZE
)

# Create validation generator
val_datagen = AugmentingDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    'val',  
    target_size=(128, 128), 
    batch_size=BATCH_SIZE 
)

# Create testing generator
test_datagen = AugmentingDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(  #'train', 'test', 'val'
    'test', 
    target_size=(128, 128), 
    batch_size=BATCH_SIZE, 
    seed=42
)

# Pick out an example
test_data = next(val_generator)
(masked, mask), ori = test_data


def plot_callback(model):
    """Called at the end of each epoch, displaying our previous test images,
    as well as their masked predictions and saving them to disk"""
    
    # Get samples & Display them        
    pred_img = model.predict([masked, mask])
    pred_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    # Clear current output and display test images
    for i in range(len(ori)):
        _, axes = plt.subplots(1, 3, figsize=(10, 5))
        axes[0].imshow(masked[i,:,:,:])
        axes[1].imshow(pred_img[i,:,:,:] * 1.)
        axes[2].imshow(ori[i,:,:,:])
        axes[0].set_title('Masked Image')
        axes[1].set_title('Predicted Image')
        axes[2].set_title('Original Image')
                
        plt.savefig(r'data/test_samples/img_{}_{}.png'.format(i, pred_time))
        plt.close()

model = Unet(vgg_weights='../pretrained_models/pytorch_to_keras_vgg16.h5')

path_models = '../pretrained_models/'
path_logs = './logs/'
# Run training for certain amount of epochs
model.fit_generator(
    train_generator, 
    steps_per_epoch=10000, 
    validation_data=val_generator,
    validation_steps=1000, 
    epochs=100,   
    verbose=0,
    callbacks=[
        TensorBoard(
            log_dir=path_logs,
            write_graph=False
        ),
        ModelCheckpoint(
            path_models+'weights.{epoch:02d}-{loss:.2f}.h5',
            monitor='val_loss', 
            save_best_only=True, 
            save_weights_only=True
        ),
        LambdaCallback(
            on_epoch_end=lambda epoch, logs: plot_callback(model)
        ),
        
    ]
)
 
# =============================================================================
#  Test using test generator
# =============================================================================

## Load weights from pretrained model
#model = Unet()
#model.load(
#    './pretrained_models/weights.97-0.95.h5',
#    train_bn=True,
#    lr=0.00005
#)
#
#n = 0
#for (masked, mask), ori in tqdm(test_generator):
#    # Run predictions for this batch of images
#    pred_img = model.predict([masked, mask])
#    pred_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#    
#    masked_image = ori* (mask)
#    # Clear current output and display test images
#    for i in range(len(ori)):
#        _, axes = plt.subplots(1, 5, figsize=(20, 5))
#        axes[0].imshow(mask[i,:,:,:])
#        axes[1].imshow(masked_image[i,:,:,:])
#        axes[2].imshow(masked[i,:,:,:] * 1.)
#        axes[3].imshow(pred_img[i,:,:,:])
#        axes[4].imshow(ori[i,:,:,:])
#        axes[0].set_title('Mask')
#        axes[1].set_title('Masked Image')
#        axes[2].set_title('CRG Image')
#        axes[3].set_title('Predicted Image')
#        axes[4].set_title('Original Image')
#                
#        plt.savefig(r'data/test_samples/img_{}_{}.png'.format(i, pred_time))
#        plt.close()
#        n += 1
#        
#    # Only create predictions for about 100 images
#    if n > 100:
#        break


 