# anchors_path = 'model_data/tiny_yolo_anchors.txt'
from keras import Input
import numpy as np
from yolo3.model import face_yolo_body


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


anchors_path = 'model_data/tiny_yolo_anchors.txt'
print(anchors_path)
anchors = get_anchors(anchors_path)
input_shape = (288, 288)


image_input = Input(shape=(288, 288, 3))
model = face_yolo_body(image_input)
model.load_weights('model_data/trained_weights_final-1-19.h5')
print("OK")

import coremltools
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.models import load_model


def model_convert(model):
    print("start converting....")
    model_coreml = coremltools.converters.keras.convert(model,
                                                        input_names=['image'],
                                                        image_input_names='image',
                                                        output_names=['output1', 'output2']
                                                        )

    model_coreml.author = 'Shimei_Zhao'
    model_coreml.short_description = 'This is a face detection keras model.'
    model_coreml.input_description['image'] = 'A 288*288 pixel image'
    model_coreml.output_description['output1'] = '9*9 of five-dimensional tensor.'
    model_coreml.output_description['output2'] = '18*18 of five-dimensional tensor.'
    model_coreml.save('logs/face_detection.mlmodel')
    print('model converted.')

from coremltools.proto import NeuralNetwork_pb2

# def convert_lambda(layer):
#     # Only convert this Lambda layer if it is for our swish function.
#     if layer.function == face_yolo_loss:
#         params = NeuralNetwork_pb2.CustomLayerParams()
#
#         # The name of the Swift or Obj-C class that implements this layer.
#         params.className = "face_yolo_loss"
#
#         # The desciption is shown in Xcode's mlmodel viewer.
#         params.description = "A fancy new activation function"
#
#         return params
#     else:
#         return None

import os.path

if os.path.isfile('model_data/trained_weights_final-1-19.h5'):
    # keras_model = load_model(model)
    model_convert(model)

else:
    print('no module found')

