from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
import sys
import numpy as np
import onnxruntime
import time

IMAGE_SIZE = 224
loop_count = 10
img_path = sys.argv[1]

# load Keras and ONNX model
net = load_model('model-resnet50-final.h5')
onnx_model = 'fish-resnet50.onnx'
sess = onnxruntime.InferenceSession(onnx_model)

# 41 + 1 classes
cls_list = ['10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
    '1', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
    '2', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
    '3', '40', '41', '42', '4', '5', '6', '7', '8', '9']

# image preprocessing
img = image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# Keras predtion
start_time = time.time()
for i in range(loop_count):
    pred = net.predict(x)[0]
print("Keras inferences with %s second in average" %((time.time() - start_time) / loop_count))

# ONNX prediction
x = x if isinstance(x, list) else [x]
feed = dict([(input.name, x[n]) for n, input in enumerate(sess.get_inputs())])

start_time = time.time()
for i in range(loop_count):
    pred_onnx = sess.run(None, feed)[0]
print("ONNX inferences with %s second in average" %((time.time() - start_time) / loop_count))

