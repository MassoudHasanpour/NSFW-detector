# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 23:59:58 2022

@author: Masoud H
"""
from flask import Flask, request
import numpy as np
import tensorflow as tf
# import pandas as pd
import tensorflow_hub as hub
# import keras
# from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from nsfw_detector import predict
import  urllib
import joblib
import jsonpickle
import numpy as np
import cv2

image_dim=299;

url = request.args.get('https://news.varzeshe3.com/pictures/2022/03/06/B/meug111j.jpg')

print('1')
urllib.request.urlretrieve(url, "local-filename.jpg")
img, image_paths = predict.load_images("images (7).jpg", (image_dim, image_dim))
nd_images=img
model_preds = model.predict(nd_images)
categories = ['Nude', 'Safe']
probs = []
print('2')
for i, single_preds in enumerate(model_preds):
    single_probs = {}
    for j, pred in enumerate(single_preds):
        single_probs[categories[j]] = float(pred)
    probs.append(single_probs)
# Results= dict(zip(image_paths, probs))
Decision = []

Decision = []
Blckd_Im = []
Cls = []
i = 0
if model_preds[i,1]>0.45:
   Cls.append([1])
   Decision.append('Block')
#       Blckd_Im.append(image_paths[i])
elif (model_preds[i,1]>0.25) and (model_preds[i, 1]<=0.45):
   Decision.append('Needs to be decided by admin')

else:
   Cls.append([0])
   Decision.append('Safe')

response = {Decision[0]}
print(model_preds)