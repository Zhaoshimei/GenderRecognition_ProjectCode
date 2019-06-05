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


anchors_path = r'C:\DL-face\YOLO_face\model_data\tiny_yolo_anchors.txt'
print(anchors_path)
anchors = get_anchors(anchors_path)
input_shape = (288, 288)


image_input = Input(shape=(None, None, 3))
model = face_yolo_body(image_input)
model.load_weights(r'C:\DL-face\YOLO_face\logs\trained_weights_final.h5')
print("OK")