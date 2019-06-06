
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from utils import get_random_wider_data
from yolo3.model import face_yolo_body, yolo_loss, face_yolo_loss, preprocess_true_face_boxes
from yolo3.utils import compose


def _main():
    log_dir = "logs/000/"
    train_annotation_path = "/home/text/ZHY/YOLO_face/dataset/wider_face_split/my_label.txt"
    val_annotation_path = "/home/text/ZHY/YOLO_face/dataset/wider_face_split/wider_face_val_bbx_gt.txt"
    anchors_path = 'model_data/tiny_yolo_anchors.txt'
    print(anchors_path)
    anchors = get_anchors(anchors_path)
    input_shape = (288, 288)  # multiple of 32, hw
    print('get it')
    model = create_face_yolo_model(input_shape,anchors,freeze_body = 2)
    print('get model')
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)


    val_split = 0.1
    with open(train_annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val


    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
    # By ny
    # model.compile(optimizer=Adam(lr=1e-3), loss={
    #     # use custom yolo_loss Lambda layer.
    #     'yolo_loss': lambda y_true, y_pred: y_pred})

        model.compile(optimizer=Adam(lr=1e-3), loss={'output1':face_yolo_loss, 'output2':face_yolo_loss})

        batch_size = 128
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_face_wrapper(lines[:num_train], batch_size, input_shape),
                            steps_per_epoch=max(1, num_train // batch_size),
                            validation_data=data_generator_face_wrapper(lines[num_train:], batch_size, input_shape),
                            validation_steps=max(1, num_val // batch_size),
                            epochs=50,
                            initial_epoch=0,
                            callbacks=[logging, checkpoint])
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        # model.compile(optimizer=Adam(lr=1e-4),
        #               loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change
        model.compile(optimizer=Adam(lr=1e-4),loss={'output1':face_yolo_loss, 'output2':face_yolo_loss})  # recompile to apply the change

        print('Unfreeze all of the layers.')

        batch_size = 32  # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_face_wrapper(lines[:num_train], batch_size, input_shape),
                            steps_per_epoch=max(1, num_train // batch_size),
                            validation_data=data_generator_face_wrapper(lines[num_train:], batch_size, input_shape),
                            validation_steps=max(1, num_val // batch_size),
                            epochs=100,
                            initial_epoch=50,
                            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_final.h5')

        # Further training if needed.

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

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
#     model_loss = Lambda(face_yolo_loss, output_shape=(1,), name='yolo_loss',
#         arguments={'ignore_thresh': 0.7})(
#         [*model_body.output, *y_true])
#     model = Model([model_body.input, *y_true], model_loss)

    return model_body



def data_generator(annotation_lines, batch_size, input_shape):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_wider_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true1,y_true2 = preprocess_true_face_boxes(box_data, input_shape)
        yield (image_data, {'output1':y_true1,'output2':y_true2})

def data_generator_face_wrapper(annotation_lines, batch_size, input_shape):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape)


if __name__ == '__main__':
    _main()
