from os.path import join, dirname, exists
import numpy as np
import cv2
from keras.models import model_from_json
from scipy.special import expit
from optparse import OptionParser

def softmax(X, theta = 1.0, axis = None):

    y = np.atleast_2d(X)

    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    y = np.exp(y * float(theta))
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    return y / ax_sum


class YOLONet(object):

    """The YOLO net built on Keras with TF as backend"""

    def __init__(self, model_name):
        """Set the model parameters"""
        work_dir = dirname(dirname(__file__))
        self.model_json_file = join(work_dir, 'model_data/{}.json'.format(model_name))
        self.model_weight_file = join(work_dir, 'model_data/{}.h5'.format(model_name))
        self.model_anchor_file = join(work_dir, 'model_data/{}_anchors.txt'.format(model_name))
        self.model_class_name_file = join(work_dir, 'model_data/{}.names'.format(model_name))
        self.model = None
        self.num_predictor = None
        self.num_classes = None
        self.detection_thresh = 0.5
        self.iou_thresh = 0.4

    def _load_model(self):
        """load the yolo model"""
        if not exists(self.model_json_file):
            print("Model {} does not exist..".format(self.model_json_file))
            return False

        json_file = open(self.model_json_file, 'r')
        loaded_model_json = json_file.read()
        yolo_model = model_from_json(loaded_model_json)
        self.model = yolo_model
        return True

    def _load_weight(self):
        """load the weight file of the model"""
        if not self.model:
            print("Invalid model to load weight file..")
            return False

        if not exists(self.model_weight_file):
            print("No weight file {} exists..".format(self.model_weight_file))
            return False

        self.model.load_weights(self.model_weight_file)
        return True

    def _load_anchor_file(self):
        """load the anchor file for prediction"""
        if not exists(self.model_anchor_file):
            print("No valid anchor file to load : {}...".format(self.model_anchor_file))
            return False

        anchor_file = open(self.model_anchor_file, 'r')
        anchors = anchor_file.readline()
        anchors = anchors.replace('\n', '')
        anchors = [float(i) for i in anchors.split(',')]
        self.anchors = np.array(anchors).reshape(-1, 2)
        self.num_predictor = len(self.anchors)
        return True

    def _load_class_name(self):
        """load the class name for detection"""
        if not exists(self.model_class_name_file):
            print("No valid name list for detection : {}..".format(self.model_class_name_file))
            return False

        name_list_file = open(self.model_class_name_file, 'r')
        class_names = name_list_file.readlines()
        self.class_names = [c.strip() for c in class_names]
        self.num_classes = len(self.class_names)
        return True

    def load_network(self):
        """load the model for detection"""

        # step 1: load the network
        if not self._load_model():
            return False

        # step 2: load the weights
        if not self._load_weight():
            return False

        # step 3: load the anchors and class names
        if not self._load_anchor_file() or not self._load_class_name():
            return False

        input_tensor = self.model.input
        input_dim = int(input_tensor.shape[2])

        output_tensor = self.model.output
        output_dim = int(output_tensor.shape[3])
        self.input_dim = input_dim

        print("Model loaded ..")

        return output_dim == (self.num_classes + 5)*self.num_predictor


    def _prep_test_image(self, image):
        """pre-process the image for testing"""

        im = cv2.resize(image, (self.input_dim, self.input_dim)).astype(np.float32)
        im = im/255
        im = np.expand_dims(im, axis=0)

        return im

    def single_image_test(self, image_file):

        if not exists(image_file):
            print("Test image {} does not exist ...".format(image_file))
            return

        image = cv2.imread(image_file)
        image = self._prep_test_image(image)
        predictions = self.model.predict(image)
        boxes, scores, classes = self.convert_prediction(predictions, self.anchors, self.num_classes)
        self.refine_display(image_file, boxes, scores, classes, display=True)

    def refine_display(self, image_file, boxes, scores, classes, display=False):

        image = cv2.imread(image_file)
        image_shape = image.shape[:2]
        height = image_shape[0]
        width = image_shape[1]
        image_dims = np.stack([width, height, width, height])
        image_dims = np.reshape(image_dims, [1, 4])
        boxes = boxes * image_dims

        nms_index = self.non_max_suppression(
            boxes, scores, iou_threshold=self.iou_thresh)
        boxes = boxes[nms_index]
        scores = scores[nms_index]
        classes = classes[nms_index]

        if not display:
            return

        # display

        for i, box in enumerate(boxes):

            pt_1 = (int(box[0]), int(box[1]))
            pt_2 = (int(box[2]), int(box[3]))
            class_name = self.class_names[classes[i]]
            print("[{}] class name {}, prob {}".format(i, class_name, scores[i]))
            cv2.rectangle(image, pt_1, pt_2, (0,255,0), 2)
            cv2.putText(image, class_name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Test', image)
        cv2.waitKey(0)

    def convert_prediction(self, feats, anchors, num_classes):

        num_sample, conv_dims, _, __ = feats.shape

        num_anchors = 5

        anchors_points = np.reshape(anchors, (num_sample, 1, 1, num_anchors, 2))

        feats = np.reshape(feats, (num_sample, conv_dims, conv_dims, num_anchors, num_classes+5))

        box_xy = expit(feats[..., :2])
        box_wh = np.exp(feats[..., 2:4])
        box_confidence = expit(feats[..., 4:5])
        box_class_probs = softmax(feats[..., 5:], axis = 4)

        conv_dims = np.shape(feats)[1:3]  # assuming channels last
        conv_height_index = np.arange(0, stop=conv_dims[0])
        conv_width_index = np.arange(0, stop=conv_dims[1])
        conv_height_index = np.tile(conv_height_index, [conv_dims[1]])
        conv_width_index = np.tile(
            np.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
        conv_width_index = np.transpose(conv_width_index)
        conv_width_index = conv_width_index.flatten()

        conv_index = np.transpose(np.stack([conv_height_index, conv_width_index]))
        conv_index = np.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
        conv_index = conv_index.astype(feats.dtype)

        box_xy = (box_xy + conv_index) / conv_dims
        box_wh = box_wh * anchors_points / conv_dims

        boxes = self.boxes_to_corners(box_xy, box_wh)

        boxes, scores, classes = self.filter_boxes(
            boxes, box_confidence, box_class_probs, threshold=self.detection_thresh)

        return boxes, scores, classes

    @staticmethod
    def boxes_to_corners(box_xy, box_wh):
        """Convert box predictions to bounding box corners."""
        box_mins = box_xy - (box_wh / 2.)
        box_maxes = box_xy + (box_wh / 2.)

        return np.concatenate([
            box_mins[..., 0:1],  # x_min
            box_mins[..., 1:2],  # y_min
            box_maxes[..., 0:1],  # x_max
            box_maxes[..., 1:2]  # y_max
        ], axis=4)

    @staticmethod
    def filter_boxes(boxes, box_confidence, box_class_probs, threshold=.24):
        """Filter boxes based on object and class confidence."""
        box_scores = box_confidence * box_class_probs
        box_classes = np.argmax(box_scores, axis=-1)
        box_class_scores = np.max(box_scores, axis=-1)
        prediction_mask = box_class_scores >= threshold

        boxes = boxes[prediction_mask]
        scores = box_class_scores[prediction_mask]
        classes = box_classes[prediction_mask]
        return boxes, scores, classes

    def non_max_suppression(self, boxes, scores, iou_threshold=0.4):
        """
        Input the bbox coordinates, scoress, and threshold,
        Output the index to keep
        """
        remove_index = []

        for i in range(len(boxes)-1):
            bbox_1 = boxes[i]
            if i in remove_index:
                continue
            for j in range(i+1, len(boxes)):
                bbox_2 = boxes[j]
                iou = self.calc_iou(bbox_1, bbox_2)
                if iou>iou_threshold:
                    r_index = i if scores[i] < scores[i] else j
                    if r_index not in remove_index:
                        remove_index.append(r_index)
        saved_index = list(set(range(len(boxes))) - set(remove_index))
        return np.array(saved_index)

    @staticmethod
    def calc_iou(bbox_1, bbox_2):
        """calculate the iou of two bounding boxes"""
        x_1_min, y_1_min, x_1_max, y_1_max = bbox_1
        x_2_min, y_2_min, x_2_max, y_2_max = bbox_2
        x_A = max(x_1_min, x_2_min)
        y_A = max(y_1_min, y_2_min)
        x_B = min(x_1_max, x_2_max)
        y_B = min(y_1_max, y_2_max)

        if x_B < x_A or y_B < y_A:
            return 0

        inter_area = (x_B - x_A + 1) * (y_B - y_A + 1)
        bbox_1_area = (x_1_max - x_1_min + 1) * (y_1_max - y_1_min + 1)
        bbox_2_area = (x_2_max - x_2_min + 1) * (y_2_max - y_2_min + 1)
        return inter_area / float(bbox_1_area + bbox_2_area - inter_area)


    def test(self, image_path):
        if not self.load_network():
            print("Unable to load the network ...")
            return
        self.single_image_test(image_path)

def main():

    parser = OptionParser()
    parser.add_option("-F", "--file", dest="filename", help="Input file name", type="string")

    options, args = parser.parse_args()

    if options.filename:
        print("Detecting in file {}".format(options.filename))
    else:
        print("No valid input file..")
        return

    yolo_net = YOLONet('yolo')
    yolo_net.test(options.filename)


if __name__ == '__main__':
    main()