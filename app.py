"""
keras==2.3.1
tensorflow==1.4.1


"""


from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# tensorflow
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import os
import glob
import shutil
import sys
import tensorflow as tf
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import glob
import shutil
import sys
import numpy as np
#from skimage.io import imread
import matplotlib.pyplot as plt
from flask import jsonify
from keras.optimizers import Adam
# Hyper Parameters
batch_size = 4
width = 224
height = 224
img_height=224
img_width=224
img_height=224
img_width=224
epochs = 50
NUM_TRAIN = 2000
NUM_TEST = 1000
dropout_rate = 0.5
input_shape = (height, width, 3)

import os
import subprocess
import sys
import keras_metrics
#import git
#git.Git().clone("https://github.com/Tony607/efficientnet_keras_transfer_learning")
#if not os.path.isdir("efficientnet_keras_transfer_learning"):
#  !git clone https://github.com/Tony607/efficientnet_keras_transfer_learning
#os.chdir('efficientnet_keras_transfer_learning/')
#%cd efficientnet_keras_transfer_learning/
#import git
#git.Git("efficientnet_keras_transfer_learning/").clone("https://github.com/Tony607/efficientnet_keras_transfer_learning")
#os.chdir("efficientnet_keras_transfer_learning/")


# Options: EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3
# Higher the number, the more complex the model is.
from keras.applications import DenseNet169

import numpy as np
from skimage.io import imread
from skimage import exposure, color
from skimage.transform import resize


import os
# Standard dependencies
import cv2
import time
import scipy as sp
import numpy as np
import random as rn
import pandas as pd
from tqdm import tqdm
from PIL import Image
from functools import partial
import matplotlib.pyplot as plt
# Standard dependencies
import cv2
import time
import scipy as sp
import numpy as np
import random as rn
import pandas as pd
from tqdm import tqdm
from PIL import Image
from functools import partial
import matplotlib.pyplot as plt
CHANNELS = 3


# Parameters
batch_size = 48

width = 224
height = 224
epochs = 20
NUM_TRAIN = 2000
NUM_TEST = 1000
dropout_rate = 0.2
input_shape = (height, width, 3)

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from tensorflow import keras
# Define a flask app
app = Flask(__name__)


session = tf.Session()

keras.backend.set_session(session)

# Model saved with Keras model.save()
# loading pretrained conv base model

modelq = models.load_model('cnn.h5',compile=False)
modelq._make_predict_function()
print('Model loaded. Start serving...')



# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


from tensorflow.keras.preprocessing import image
'''
def predict_image(img_path):
    output=[]
    img = image.load_img(img_path, target_size=(height, width))
                # Convert it to a Numpy array with target shape.
    x = image.img_to_array(img)
                # Reshape
    x = x.reshape((1,) + x.shape)
    x /= 255.
    classes = modelq.predict_classes([x])
    result = modelq.predict([x])[0][0]

    if classes.item(0) == 0:
        imagetype = "No ROP"
    else:
        imagetype = "Yes ROP"
        result = 1 - result
    output.append(imagetype)
    output.append(str(classes.item(0)))
    output.append(str(result))
    dict={'ROP':str(output[0]),"Class":str(output[1]),"Probability Score":str(output[2])}
    return ('Prediction is: '+str(output[0]) +" "+ 'Image Class: '+str(output[1]) +" "+ 'Probability Score: '+str(output[2]))

'''
def predict_image(img_path):
    print("im at predict image ***************************")
    output=[]
    try:
        with session.as_default():
            with session.graph.as_default():
                # Read the image and resize it
                img = image.load_img(img_path, target_size=(height, width))
                # Convert it to a Numpy array with target shape.
                x = image.img_to_array(img)
                # Reshape
                x = x.reshape((1,) + x.shape)
                x /= 255.
                classes = modelq.predict_classes([x])
                result = modelq.predict([x])[0][0]
                print(result)
                if result > 0.5:
                    imagetype = "Covid"
                else:
                    imagetype = "Normal"
                    result = 1 - result
                output.append(imagetype)
                output.append(str(classes.item(0)))
                output.append(str(result))
    except Exception as ex:
        log.log('Seatbelt Prediction Error', ex, ex.__traceback__.tb_lineno)
    dict={'ROP':str(output[0]),"Class":str(output[1]),"Probability Score":str(output[2])}
    #return ('Prediction is: '+str(output[0]) +" "+ 'Image Class: '+str(output[1]) +" "+ 'Probability Score: '+str(output[2]))
    return dict

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = predict_image(file_path)
        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result = str(pred_class[0][0][1])               # Convert to string
        return jsonify(preds)
    return None


if __name__ == '__main__':
    app.run(debug=True)
