### Iterative Facial Image Inpainting
This repository contains source codes and training sets for the following paper:

"Iterative Facial Image Inpainting Based on an Encoder-Generator Architecture" *by Yahya DOGAN and Hacer YALIM KELES*.

The preprint version of the above paper is available at: https://arxiv.org/abs/2101.07036

![Proposed model Architecture](https://github.com/yahyadogan72/iterative_facial_image_inpainting/blob/main/images/method.PNG?raw=true)
![Results](https://github.com/yahyadogan72/iterative_facial_image_inpainting/blob/main/images/figure%201.PNG?raw=true)

### Requirements
- Spyder 3.3.x (recommended)
- Python 3.6
- Keras 2.2.4
- Tensorflow-gpu 1.12.0

### Usage

1. Clone this repo: git clone https://github.com/yahyadogan72/iterative_facial_image_inpainting.git
2. Download [CelebA](https://github.com/tkarras/progressive_growing_of_gans/) dataset, then put it in the following directory structure:  
    |-Iterative_facial_image_inpainting  
    |---Discriminator  
    |----dataset/real
3. Prepare the dataset for the discriminator network using the [script](https://github.com/yahyadogan72/iterative_facial_image_inpainting/blob/main/Discriminator/create_dataset_for_discriminator.py/).
4. Train the discriminator network using the [script](https://github.com/yahyadogan72/iterative_facial_image_inpainting/blob/main/Discriminator/train_disriminator.py).
5. [Download](https://drive.google.com/file/d/1TmAsaZ0uCtUPCu024S7OIFw7Nz_hCz1z/view?usp=sharing) the pre-trained models and put them in the pretrained _models folder.
6. Create dataset for the Unet model using the CRG model, then put it in the following directory structure:    
   |-Iterative_facial_image_inpainting  
   |--Unet  
   |---dataset  
   |----train/input
7. Train the Unet network using the [script](https://github.com/yahyadogan72/iterative_facial_image_inpainting/blob/main/Unet/train.py).
8. Test the overall model using the [script](https://github.com/yahyadogan72/iterative_facial_image_inpainting/blob/main/test_model.py).
9. You can use the [script](https://github.com/yahyadogan72/iterative_facial_image_inpainting/blob/main/mask_landmarks.py) to create a mask that includes all facial landmarks in a face or download test data [here](https://drive.google.com/file/d/1jUg9ELrbvYDr82LucFq-0WvGeDixNMl9/view?usp=sharing).
### Citation
If you find iterative facial image inpainting method useful in your research, please consider citing:
```
Dogan, Y. & Keles, H.Y. Neural Comput & Applic (2022). https://doi.org/10.1007/s00521-022-06987-y
```
### Preprint:
```
@article{dogan2021iterative,
  title={Iterative Facial Image Inpainting using Cyclic Reverse Generator},
  author={Dogan, Yahya and Keles, Hacer Yalim},
  journal={arXiv preprint arXiv:2101.07036},
  year={2021}
}
```
### Contact 
```
yahyadogan72 at gmail.com
```
