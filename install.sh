# Install the YOLO based detection in Keras (YDK)
# Download the necessary configuration files and the weight file
# The network is defined in yolo.json
# The weight file is stored in yolo.h5
# Class names can be found in yolo.names.

echo "Start to install YDK"

mkdir -p model_data

cd model_data

# 1, anchor file

wget "https://www.dropbox.com/s/mxw41tlxn091z4c/yolo_anchors.txt"

# 2, class name

wget "https://www.dropbox.com/s/0tpvaaetoxbgzpm/yolo.names"

# 3, network config

wget "https://www.dropbox.com/s/tb2541xqxus3j5k/yolo.json"

# 4, weight file

wget "https://www.dropbox.com/s/lz8dfrft3uttv8h/yolo.h5"

echo "Done"
