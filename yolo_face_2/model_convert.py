import coremltools
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.models import load_model

def model_convert(model):
    print("start converting....")
    model_coreml = coremltools,converters.keras.convert(model,
            input_names=['image'], image_input_names='image')
    
    model_coreml.author = 'Shimei_Zhao'
    model_coreml.short_description = 'This is a face detection keras model.'
    model_coreml.input_description['image'] = 'A 288*288 pixel image'
    model_coreml.output_description['output'] = '9*9 of five-dimensional tensor.'
    model_coreml.save('face_detection.mlmodel')
    print('model converted.')

import os.path
if os.path.isfile('trained_weights_final.h5'):
    keras_model = load_model('trained_weights_final.h5')
    model_convert(keras_model)

else:
    print('no module found')
