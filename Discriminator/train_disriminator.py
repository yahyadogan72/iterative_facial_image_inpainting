# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 14:37:52 2020

@author: user
"""
from __future__ import print_function
import numpy as np
from keras.preprocessing import image
import os
import keras
from keras.layers import Dense, Flatten, Input, Conv2D,MaxPooling2D, BatchNormalization, SpatialDropout2D
from keras.models import Model#, load_model
from keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
import random as rn
from IPython.display import clear_output

# =============================================================================
#  For reproducible results
# =============================================================================
np.random.seed(42)
rn.seed(12345)

vgg_weights_path = '../pretrained_models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
path_model = '../pretrained_models/'
path_log = './log/'
path = './dataset/'

opt = keras.optimizers.RMSprop(lr = 0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
val_split = 0.2
max_epochs = 300
batch_size = 128
dropout_rate = 0.5

# =============================================================================
#  Dataloader
# =============================================================================
def get_images_array_and_label(path):
    # label = 0
    label = 0.1
    iterr = 0
    images = np.zeros([60000, 128,128,3])
    labels = np.zeros([60000])
    
    # =============================================================================
    #  Create two folders under the dataset folder. These are:
    #       1. Fake images (30k images, which are highly distorted, are labeled as fake.)
    #       2. Real images (22k of these images are undistorted images and 8k of them are mildly distorted, both of which are labeled as real)    
    # =============================================================================
    for mainfolder in os.listdir(path):
#        if mainfolder == 'masks' or mainfolder == 'reals':
#            break
        main_folder_path = os.path.join(path, mainfolder)
#        print(main_folder_path)
        for image_id in os.listdir(main_folder_path):
            # print(image_id)
            path_image = os.path.join(main_folder_path, image_id)
            img = image.load_img(path_image, target_size=(128, 128,3))
            img = image.img_to_array(img)
            img = preprocess_input(img)
            labels [iterr] = label
            images[iterr]= img
            iterr +=1
#        label +=1
        label +=0.8
    return images, labels  

images, labels = get_images_array_and_label(path)


# =============================================================================
#  Model
# =============================================================================

def VGG16(x):
    
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(x)
#    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(rate=dropout_rate, name='block1_dr2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
#    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(rate=dropout_rate, name='block2_dr2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(rate=dropout_rate, name='block3_dr3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
     
    # Block 4
#    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
#    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
#    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
#    x = BatchNormalization()(x)
#    x = SpatialDropout2D(rate=dropout_rate, name='block4_dr1')(x)
     
    # Block 6
#    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv1')(x)
#    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv2')(x)
#    x = BatchNormalization()(x)
#    x = SpatialDropout2D(rate=dropout_rate, name='block6_dr3')(x)
#    x = Conv2D(1024, (3, 3), activation='relu', padding='same', name='block6_conv3')(x)
#    x = Conv2D(1024, (3, 3), activation='relu', padding='same', name='block6_conv4')(x)
#    x = BatchNormalization()(x)
#    x = SpatialDropout2D(rate=dropout_rate, name='block6_dr4')(x)
      
    return x
    
def Discriminator():
     inputs = Input(shape=(128,128,3),name = 'image_input')
     vgg_layer_output = VGG16(inputs)
     
     x = Flatten()(vgg_layer_output)
     x = Dense(1, activation="sigmoid", name='dense_layer')(x)

     model = Model(inputs=inputs, outputs=x, name='model')
     model.load_weights(vgg_weights_path, by_name=True)
#     model.summary()
#     
#     freeze_layers = ['block1_conv1','block1_conv2', 'block2_conv1', 'block2_conv2', 'block3_conv1', 'block3_conv2', 'block3_conv3']
##     freeze_layers = ['block1_conv1','block1_conv2', 'block2_conv1', 'block2_conv2']
#     for layer in model.layers:
#        if(layer.name  in freeze_layers):
#            layer.trainable = False
     
     return model
model = Discriminator()
#model=load_model(model_path+'Discriminator.h5')
#model.summary()

# =============================================================================
#  Loss plot callback
# =============================================================================
  
class Loss_plot_callback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();
        
Loss_plot_callback = Loss_plot_callback()

# =============================================================================
#  Training
# ============================================================================= 

model.compile(loss=keras.losses.binary_crossentropy, optimizer=opt)


model_path = os.path.join(path_model, 'Discriminator' +  '.h5')

chk = keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', verbose=0, 
                                      save_best_only=True, save_weights_only=False, mode='auto', period=1)

redu = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, verbose=1, mode='auto')#monitor='val_loss'

#early = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=20, verbose=0, mode='auto')
#
tensor_brd = keras.callbacks.TensorBoard(log_dir=path_log, histogram_freq=0, batch_size=batch_size, write_graph=True,
                                         write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, 
                                         embeddings_metadata=None, embeddings_data=None, update_freq='epoch')


history= model.fit(images, labels, validation_split=val_split, epochs=max_epochs, batch_size=batch_size, 
                   callbacks=[chk, redu,Loss_plot_callback], verbose=1, class_weight=None, shuffle = True)

#history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
#	validation_data=(x_val, y_val), steps_per_epoch=len(x_train) // batch_size, callbacks=[chk, redu, tensor_brd, Loss_plot_callback],
#	epochs=max_epochs, shuffle=True)