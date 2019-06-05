from keras.engine.saving import load_model
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model

from yolo3.model import face_yolo_body, yolo_loss, face_yolo_loss, preprocess_true_face_boxes

def create_face_yolo_model(input_shape, anchors,load_pretrained=True, freeze_body=2,
            weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    # calculate which grid that true ob belongs to   2 feature maps
    # [[9 * 9 * 5], [18 * 18 * 5]]
    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], 5)) for l in range(2)]
    # y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l],num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = face_yolo_body(image_input)
    # print('Create Tiny face YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
    model_loss = Lambda(face_yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

# anchors_path = 'model_data/tiny_yolo_anchors.txt'
anchors_path = r'C:\DL-face\YOLO_face\model_data\tiny_yolo_anchors.txt'
print(anchors_path)
anchors = get_anchors(anchors_path)
input_shape = (288, 288)
model = create_face_yolo_model(input_shape,anchors,load_pretrained=False,freeze_body = 0)
model.load_weights(r'C:\Users\NY\Desktop\trained_weights_final.h5')
print("OK")