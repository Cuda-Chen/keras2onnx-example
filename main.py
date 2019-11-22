from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
import sys
import numpy as np
import keras2onnx
import onnxruntime

IMAGE_SIZE = 224

img_path = sys.argv[1]

#net = load_model('model-resnet50-final.h5')
#onnx_net = keras2onnx.convert_keras(net, net.name)
onnx_model = 'fish-resnet50.onnx'

# 41 + 1 classes
cls_list = ['10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
    '1', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
    '2', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
    '3', '40', '41', '42', '4', '5', '6', '7', '8', '9']

# image preprocessing
img = image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# runtime prediction
#content = onnx_net.SerializeToString()
#sess = onnxruntime.InferenceSession(content)
sess = onnxruntime.InferenceSession(onnx_model)
x = x if isinstance(x, list) else [x]
feed = dict([(input.name, x[n]) for n, input in enumerate(sess.get_inputs())])
pred_onnx = sess.run(None, feed)[0]
pred = np.squeeze(pred_onnx)
top_inds = pred.argsort()[::-1][:5]
print(img_path)
for i in top_inds:
    print('    {:.3f}  {}'.format(pred[i], cls_list[i]))

