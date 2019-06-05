# anchors_path = 'model_data/tiny_yolo_anchors.txt'
from keras import Input
import numpy as np
from yolo3.model import face_yolo_body
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import os



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



def load_image(img_path, show=False):
    
    img = image.load_img(img_path, target_size=(288, 288))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]
    
    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()
    
    return img_tensor


# image path
img_path = 'dog416.png'

# load a single image
new_image = load_image(img_path)

# check prediction
pred = model.predict(new_image)
print(pred[0].shape)
grid_9_pred = pred[0]
#print("the prediction result:", pred)
for cy in range(9):
    for cx in range(9):
        tx = grid_9_pred[0,cy, cx, 0]
        ty = grid_9_pred[0,cy, cx, 1]
        tw = grid_9_pred[0,cy, cx, 2]
        th = grid_9_pred[0,cy, cx, 3]
        tc = grid_9_pred[0,cy, cx, 4]
#        print(tx,ty,tw,th,tc)
