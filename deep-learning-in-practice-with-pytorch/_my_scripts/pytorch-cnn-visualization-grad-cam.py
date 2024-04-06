# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 13:14:51 2024

@author: Alberto
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torchvision

from PIL import Image, ImageFilter

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from torchvision.models import vgg16, VGG16_Weights
from torchvision.transforms import v2

model = vgg16(weights=VGG16_Weights.DEFAULT)
target_layers = [ model.features[-1] ]

# load image
image_url = "https://api.time.com/wp-content/uploads/2014/07/492290913.jpg"
image_raw = Image.open(requests.get(image_url, stream=True).raw)
image_width, image_height = image_raw.size
print("Original image size:", image_raw.size)

cropped_image = image_raw.crop((image_width-image_height, 0, image_width, image_height))
print("Size of the cropped image:", cropped_image.size)

resized_image = cropped_image.resize((224, 224))

# now, we have to separately prepare a resized 'visualizable' image as a numpy array
# because pytorch_grad_cam does only likes images with a certain format
visualizible_image = np.float32(resized_image) / 255

# preprocess image for tensor
imagenet_composed_transformation = torchvision.transforms.Compose(
                        [
                            v2.PILToTensor(),
                            # here, it is cropping using the smallest dimension
                            v2.CenterCrop(min(image_height, image_width)), 
                            v2.Resize((224, 224)),
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Normalize(mean=(0.485, 0.456, 0.406), 
                                        std=(0.229, 0.224, 0.225))
                        ]
                        )
image_tensor = imagenet_composed_transformation(resized_image)

# prepare the gradcam object
cam = GradCAM(model=model, target_layers=target_layers)

# prepare target
targets = [ ClassifierOutputTarget(281) ]

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=image_tensor.unsqueeze(0), targets=targets)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
cam_image = show_cam_on_image(visualizible_image, grayscale_cam, use_rgb=True)

fig, ax = plt.subplots()
ax.imshow(cam_image)


# You can also get the model outputs without having to re-inference
model_outputs = cam.outputs