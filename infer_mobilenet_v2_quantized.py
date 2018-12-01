import numpy as np
import skimage.io
import skimage.transform
import operator
import os
from caffe2.python import workspace, core, model_helper
from caffe2.proto import caffe2_pb2
from helpers import *

##### Load the Model
# Load the pre-trained model

init_net = caffe2_pb2.NetDef()
predict_net = caffe2_pb2.NetDef()

# read .pb files

filename = "mobilenet_v2_quantized/init_net.pb"
print("Reading " + filename)
with open(filename, 'rb') as f:
    init_net.ParseFromString(f.read())

filename = "mobilenet_v2_quantized/predict_net.pb"
print("Reading " + filename)
with open(filename, 'rb') as f:
    predict_net.ParseFromString(f.read())
# Initialize the predictor with Mynet's init_net and predict_net
p = workspace.Predictor(init_net, predict_net)

##### Select and format the input image
# use whatever image you want (urls work too)
# img = "https://upload.wikimedia.org/wikipedia/commons/a/ac/Pretzel.jpg"
# img = "images/cat.jpg"
# img = "images/cowboy-hat.jpg"
# img = "images/cell-tower.jpg"
# img = "images/Ducreux.jpg"
# img = "images/pretzel.jpg"
# img = "images/orangutan.jpg"
# img = "images/aircraft-carrier.jpg"
img = "flower.jpg"

# average mean to subtract from the image
mean = 128
# the size of images that the model was trained with
input_size = 224

# use the image helper to load the image and convert it to NCHW
img = loadToNCHW(img, mean, input_size)

##### Run the test
# submit the image to net and get a tensor of results
results = p.run({'data': img})

##### Process the results
# Quick way to get the top-1 prediction result
# Squeeze out the unnecessary axis. This returns a 1-D array of length 1000
preds = np.squeeze(results)
# Get the prediction and the confidence by finding the maximum value and index of maximum value in preds array
curr_pred, curr_conf = max(enumerate(preds), key=operator.itemgetter(1))
print("Top-1 Prediction: {}".format(curr_pred))
print("Top-1 Confidence: {}\n".format(curr_conf))

# Lookup our result from the inference list
response = parseResults(results)
print(response)
