# keras2onnx-example
Keras to ONNX Example.

# How to Set up and Run
1. Download the pre-trained model from [here](https://drive.google.com/open?id=1ouJ8xZzi6x2cEkojS3DC1Wy77zjBGP1c)
2. Type the following commands for setting up and run:
```
$ conda create -n keras2onnx-example python=3.6 pip
$ conda activate keras2onnx-example
$ pip install -r requirements.txt
$ python convert_keras_to_onnx.py
$ python main.py 3_001_0.bmp
```
And it should output the following messages in the end:
```
...
3_001_0.bmp
    1.000  3
    0.000  37
    0.000  42
    0.000  14
    0.000  17
```
