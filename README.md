# YDK
 YOLO based object detection in Keras. The network config and the weight file are downloaded from the original [Darknet](https://pjreddie.com/darknet/yolo/). Network config and weight file are converted to Keras compatible JSON file and H5 file respectively. 

 ## Dependencies

- Python2.7
- TensorFlow 1.0 
- Keras 2.0
- Numpy
- Opencv 2.4

## Installation

```bash
git clone https://github.com/qiningonline/YDK.git
cd YDK
sh install.sh
```

## Run detection

```bash
python ./src/test_yolo.py ./test/dog.jpg
```

## Reference

- [Darkket](https://pjreddie.com/darknet/yolo/)

- [YAD2K](https://github.com/allanzelener/YAD2K) 

This repo is developed based on the work of YAD2K. Different from YAD2K, this work removed the TF dependencies in detection.
