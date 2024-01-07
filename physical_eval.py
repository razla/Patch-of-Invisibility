import time

from pytorchYOLOv4.demo import DetectorYolov4
from pytorchYOLOv4.demo import detect_cv2_camera
from tool.darknet2pytorch import Darknet
from pytorchYOLOv4.tool.torch_utils import *
import numpy as np
import math

import gluoncv as gcv
from gluoncv.utils import try_import_cv2
import matplotlib.pyplot as plt
cv2 = try_import_cv2()
import mxnet as mx

def plot_boxes_cv2(img, boxes, savename=None, class_names=None, color=None):
    import cv2
    img = np.copy(img)
    colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)

        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            if color is None:
                rgb = (red, green, blue)
            img = cv2.putText(img, class_names[cls_id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 1)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 1)
    if savename:
        print("save plot results to %s" % savename)
        cv2.imwrite(savename, img)
    return img

def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names

class_names = load_class_names('/Users/razlapid/PhD/Naturalistic-Adversarial-Patch/coco.names')
model = 'yolov3'
# Load the model
if model == 'yolov3':
    net = gcv.model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
else:
    net = Darknet(cfgfile='/Users/razlapid/Downloads/yolov4-tiny.cfg')
# Compile the model for faster speed
# net.hybridize()

# Load the webcam handler
cap = cv2.VideoCapture(0)
time.sleep(1) ### letting the camera autofocus

axes = None
NUM_FRAMES = 200 # you can change this
for i in range(NUM_FRAMES):
    # Load frame from the camera
    ret, frame = cap.read()


    # Image pre-processing
    frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
    if model == 'yolov3':
        rgb_nd, frame = gcv.data.transforms.presets.ssd.transform_test(frame, short=512, max_size=700)
    else:
        rgb_nd, frame = gcv.data.transforms.presets.ssd.transform_test(frame, short=416, max_size=416)

    # Run frame through network
    if model == 'yolov3':
        class_IDs, scores, bounding_boxes = net(rgb_nd)
    else:
        start = time.time()
        boxes = do_detect(net, torch.tensor(rgb_nd.asnumpy()), 0.4, 0.6, use_cuda=False)
        finish = time.time()
        print('Predicted in %f seconds.' % (finish - start))

    if model == 'yolov3':
        img = gcv.utils.viz.cv_plot_bbox(frame, bounding_boxes[0], scores[0], class_IDs[0], class_names=net.classes)
    else:
        img = gcv.utils.viz.cv_plot_bbox(frame, boxes[0], [], [], class_names=['0', '1'])
    result_img = plot_boxes_cv2(img, boxes[0], savename=None, class_names=class_names)

    cv2.imshow('Yolo demo', result_img)
    cv2.waitKey(1)

    # Display the result
    gcv.utils.viz.cv_plot_image(img)


    if i % 5 == 0:
        plt.imshow(img)
        plt.savefig(f'img_{i}.png')
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()