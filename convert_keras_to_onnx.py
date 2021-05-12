from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
import onnx
import os
os.environ['TF_KERAS'] = '1'
import keras2onnx

onnx_model_name = 'fish-resnet50.onnx'

model = load_model('model-resnet50-final.h5')
onnx_model = keras2onnx.convert_keras(model, model.name)
onnx.save_model(onnx_model, onnx_model_name)
