# GenderRecognition_ProjectCode
This is for storing the code for the Computer Science project
# Gender Recognition Project

This is the relevant code for developing a gender recognition iOS application using deep learning methods. During this process, there are mainly involve three parts relating to coding work.

## Face Detection Model

The files in "Face_Yolo" are the main files that are used to obtain the improved model based on TinyYoloV3 algorithm. It contains all the experiment code for model design, data processing, model training, testing etc.

To obtain the desirable model, it refers to some of the resources from: <https://github.com/qqwweee/keras-yolo3> 

To compare the results of Faster-RCNN algorithm with improved model, it refers to some of the work from: <https://github.com/jinfagang/keras_frcnn>

The dataset used for training the model used is cited from: <http://shuoyang1213.me/WIDERFACE/>

## Gender Recogniiton Model

The gender recognition is build on full CNN network, which refers to the part of the work from: <https://talhassner.github.io/home/publication/2015_CVPR>

The dataset used improving the model performance on different ethnicities sources from: <http://afad-dataset.github.io/>

## Application Design
The code for the real-time detection project is saved in file "Face_Yolo".
It refers to part of the work from: <https://github.com/hollance/YOLO-CoreML-MPSNNGraph>

The code for the pictture mode detection is saved in file "Face_Yolo_Picture".
It refers to paert of the work from: <https://github.com/ph1ps/Food101-CoreML>
