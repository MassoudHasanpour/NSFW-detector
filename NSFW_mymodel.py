# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 14:31:25 2022

@author: hasan
"""

# -*- coding: utf-8 -*-
from flask import Flask, request
import numpy as np
import tensorflow as tf
# import pandas as pd
import tensorflow_hub as hub
# import keras
# from keras.models import load_model
from nsfw_detector import predict
import  urllib
import joblib
import jsonpickle
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_dim=299;


app = Flask(__name__)

@app.route('/')
def hello_world():
    """
        this is just for testing server.
        send request to 127.0.0.1:5050
        you must see 'hello world'
    """
    return 'Hello World!'

model = tf.keras.models.load_model('.\model_3')
valid_datagen = ImageDataGenerator(rescale=1./255)  # no augmentation for validation set

@app.route('/n')
def NSFW_Fun():
    print('0')
    url = request.args.get('text')

    print('1')
    urllib.request.urlretrieve(url, "local-filename.jpg")
    img, image_paths = predict.load_images("images (7).jpg", (image_dim, image_dim))
    # nd_images = valid_datagen.flow_from_directory(".\local-filename.jpg",
    #                                                          batch_size=1,
    #                                                          class_mode='categorical',
    #                                                          target_size=(299, 299))
    nd_images=img/255
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
    return Decision[0]
 #   response_pickled = jsonpickle.encode(response)

#    return Decision # response_pickled # Response(response=response_pickled, status=200, mimetype="applica>
# host="0.0.0.0", port=.....
if __name__ == '__main__':
    app.run()

# end = timer()
# print(timedelta(seconds=end-start))

# np.savetxt('model_preds_S.csv', model_preds, delimiter=';')


