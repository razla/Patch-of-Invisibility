from models import *
from utils.utils import *
import argparse
from PIL import ImageDraw, ImageFont
import matplotlib.pyplot as plt

import os
import sys
import time
import datetime
import argparse

from PIL import Image

sys.path.append('PyTorchYOLOv3/')
from models import *
from utils.utils import *


class MaxProbExtractor(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, cls_id=0, tgt_cls_id=1, num_cls=80):
        super(MaxProbExtractor, self).__init__()
        self.cls_id = cls_id
        self.cls_id_tgt = tgt_cls_id
        self.num_cls = num_cls

    def set_cls_id_attacked(self, cls_id):
        self.cls_id = cls_id

    def set_cls_id_tgt(self, cls_id_tgt):
        self.cls_id_tgt = cls_id_tgt

    def forward(self, YOLOoutput):
        ## YOLOoutput size : torch.Size([4, 2535, 85])
        batch = YOLOoutput.size()[0]
        output_objectness = YOLOoutput[:, :, 4]
        output_class = YOLOoutput[:, :, 5:(5 + self.num_cls)]
        output_class_new = output_class[:, :, self.cls_id]
        output_class_tgt = output_class[:, :, self.cls_id_tgt]
        return output_objectness, output_class_new, output_class_tgt


class DetectorYolov3():
    def __init__(self, cfgfile="PyTorchYOLOv3/config/yolov3.cfg", weightfile="PyTorchYOLOv3/weights/yolov3.weights",
                 show_detail=False, tiny=False, epoch_save=-1, multi_score=True):
        #
        start_t = time.time()

        if (tiny):
            cfgfile = "PyTorchYOLOv3/config/yolov3-tiny.cfg"
            weightfile = "PyTorchYOLOv3/weights/yolov3-tiny.weights"

        self.show_detail = show_detail
        # check whether cuda or cpu
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Set up model
        self.model = Darknet(cfgfile)
        self.multi_score = multi_score
        self.model.load_darknet_weights(weightfile)
        self.img_size = self.model.img_size
        self.epoch_save = epoch_save

        print(f'Loading Tiny-YOLOv3 weights from %s... Done!' % (weightfile)) if tiny else print(
            f'Loading YOLOv3 weights from %s... Done!' % (weightfile))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            self.use_cuda = True
            self.model.cuda()
            # init MaxProbExtractor
            self.max_prob_extractor = MaxProbExtractor().cuda()
        else:
            self.use_cuda = False
            # init MaxProbExtractor
            self.max_prob_extractor = MaxProbExtractor()

        finish_t = time.time()
        if self.show_detail:
            print('Total init time :%f ' % (finish_t - start_t))

    def detect(self, input_imgs, cls_id_attacked, epoch, clear_imgs=None, with_bbox=True):
        start_t = time.time()
        # resize image
        # input_imgs = F.interpolate(input_imgs, size=self.img_size).to(self.device)
        input_imgs = input_imgs.permute(2, 0, 1) / 255.0

        # Get detections
        with torch.no_grad():
            self.model.eval()
            detections = self.model(input_imgs.unsqueeze(0))  ## v3tiny:torch.Size([8, 2535, 85]), v3:torch.Size([8, 10647, 85])
        if not (detections[0] == None):
            # init cls_id_attacked
            self.max_prob_extractor.set_cls_id_attacked(cls_id_attacked)
            # max_prob_obj, max_prob_cls = self.max_prob_extractor(detections)
            output_objectness, output_class, output_class_target = self.max_prob_extractor(detections)
            # print(output_objectness.shape, output_class.shape)

            if self.multi_score:
                output_cls_obj = output_objectness * output_class  # output_objectness = output_class = (8,845)
            else:
                output_cls_obj = output_objectness
            max_prob_obj_cls, max_prob_obj_cls_index = torch.max(output_cls_obj, dim=1)
            # print(max_prob_obj_cls.shape)
            if with_bbox and (epoch % self.epoch_save == 0):
                c_detections = detections.clone()
                bboxes = non_max_suppression(c_detections, 0.3, 0.4)  ## <class 'list'>.
                # only non None. Replace None with torch.tensor([])
                bboxes = [torch.tensor([]) if bbox is None else bbox for bbox in bboxes]
                bboxes = [rescale_boxes(bbox, self.img_size, [1, 1]) if bbox.dim() == 2 else bbox for bbox in
                          bboxes]  # shape [1,1] means the range of value is [0,1]
                # print("bboxes size : "+str(len(bboxes)))
                # print("bboxes      : "+str(bboxes))

            # get overlap_score
            if not (clear_imgs == None):
                # resize image
                input_imgs_clear = F.interpolate(clear_imgs, size=self.img_size).to(self.device)
                # detections_tensor
                detections_clear = self.model(
                    input_imgs_clear)  ## v3tiny:torch.Size([8, 2535, 85]), v3:torch.Size([8, 10647, 85])
                if not (detections_clear[0] == None):
                    #
                    # output_score_clear = self.max_prob_extractor(detections_clear)
                    output_score_obj_clear, output_score_cls_clear, output_class_target_clear = self.max_prob_extractor(
                        detections_clear)
                    output_cls_obj = output_score_obj_clear * output_score_cls_clear
                    # st()
                    output_score_clear, output_score_clear_index = torch.max(output_cls_obj, dim=1)
                    # count overlap
                    output_score = max_prob_obj_cls
                    # output_score_clear = (max_prob_obj_clear * max_prob_cls_clear)
                    overlap_score = torch.abs(output_score - output_score_clear)
                else:
                    overlap_score = torch.tensor(0).to(self.device)
            else:
                overlap_score = torch.tensor(0).to(self.device)
        else:
            print("None : " + str(type(detections)))
            print("None : " + str(detections))
            max_prob_obj = []
            max_prob_cls = []
            bboxes = []
            overlap_score = torch.tensor(0).to(self.device)

        finish_t = time.time()
        if self.show_detail:
            print('Total init time :%f ' % (finish_t - start_t))

        return max_prob_obj_cls, output_class_target, overlap_score, bboxes

def plot_boxes_cv2(img, boxes, savename=None, class_names=None, color=None, fps=None):
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

        # left, top, right, bottom

        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)

        # print(x1, y1, x2, y2)

        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            # obj_conf = box[4]
            cls_id = box[6]
            # print('%s: %f' % (class_names[int(cls_id)], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            if color is None:
                rgb = (red, green, blue)
            img = cv2.putText(img, class_names[int(cls_id)] + f"({cls_conf*100:.2f}%)", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 2)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 1)
    if savename:
        print("save plot results to %s" % savename)
        cv2.imwrite(savename, img)
    cv2.putText(img, str(fps), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
    return img



def nms_cpu(boxes, confs, nms_thresh=0.6, min_mode=False):
    # print(boxes.shape)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


def post_processing(img, conf_thresh, nms_thresh, output, show_detail=False):
    # anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
    # num_anchors = 9
    # anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    # strides = [8, 16, 32]
    # anchor_step = len(anchors) // num_anchors

    # box_array, [batch, num, 1, 4]       , torch.Size([1, 22743, 1, 4]), torch.Size([1, 2535, 1, 4])
    box_array = output[:, :, 0:4].unsqueeze(2)
    # confs,     [batch, num, num_classes], torch.Size([1, 22743, 80]), torch.Size([1, 2535, 85])
    confs = output[: , :, 5:85]

    t1 = time.time()

    if type(box_array).__name__ != 'ndarray':
        box_array = box_array.cpu().detach().numpy()
        confs = confs.cpu().detach().numpy()

    num_classes = confs.shape[2]

    # [batch, num, 4]
    box_array = box_array[:, :]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)

    t2 = time.time()

    bboxes_batch = []
    for i in range(box_array.shape[0]):

        argwhere = max_conf[i] > conf_thresh
        print(argwhere.shape)
        l_box_array = box_array[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]

        bboxes = []
        # nms for each class
        for j in range(num_classes):

            cls_argwhere = l_max_id == j
            ll_box_array = l_box_array[cls_argwhere, :]
            ll_max_conf = l_max_conf[cls_argwhere]
            ll_max_id = l_max_id[cls_argwhere]

            keep = nms_cpu(ll_box_array.squeeze(1), ll_max_conf, nms_thresh)

            if (keep.size > 0):
                ll_box_array = ll_box_array[keep, :]
                ll_max_conf = ll_max_conf[keep]
                ll_max_id = ll_max_id[keep]

                for k in range(ll_box_array.shape[0]):
                    bboxes.append(
                        [ll_box_array[k,:, 0], ll_box_array[k,:, 1], ll_box_array[k,:, 2], ll_box_array[k,:, 3], ll_max_conf[k],
                         ll_max_conf[k], ll_max_id[k]])

        bboxes_batch.append(bboxes)

    t3 = time.time()

    if (show_detail):
        print('-----------------------------------')
        print('       max and argmax : %f' % (t2 - t1))
        print('                  nms : %f' % (t3 - t2))
        print('Post processing total : %f' % (t3 - t1))
        print('-----------------------------------')

    return bboxes_batch


def do_detect(model, img, conf_thresh, nms_thresh, use_cuda=1, show_detail=False):
    model.model.eval()
    t0 = time.time()

    if type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
        img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    elif type(img) == np.ndarray and len(img.shape) == 4:
        img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)
    elif type(img) == torch.Tensor and len(img.size()) == 4:
        img = img
    else:
        print(type(img))
        print("unknow image type")
        exit(-1)

    if use_cuda:
        img = img.cuda()
    img = torch.autograd.Variable(img)

    t1 = time.time()

    output = model.model(img)

    t2 = time.time()

    if (show_detail):
        print('-----------------------------------')
        print('           Preprocess : %f' % (t1 - t0))
        print('      Model Inference : %f' % (t2 - t1))
        print('-----------------------------------')

    return post_processing(img, conf_thresh, nms_thresh, output)

def detect_cv2_camera(cfgfile, weightfile):
    import cv2
    m = DetectorYolov3(cfgfile, weightfile)

    cap = cv2.VideoCapture(0)

    prev_frame_time = 0

    # cap = cv2.VideoCapture("./test.mp4")
    cap.set(3, 1280)
    cap.set(4, 720)
    print("Starting the YOLO loop...")

    class_names = load_classes('data/coco.names')  # Extracts class labels from file
    counter = 0
    while True:
        ret, img = cap.read()
        sized = cv2.resize(img, (m.img_size, m.img_size))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        new_frame_time = time.time()

        # Calculating the fps

        # fps will be number of frame processed in given time frame
        # since their will be most of time error of 0.001 second
        # we will be subtracting it to get more accurate result
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # converting the fps into integer
        fps = int(fps)

        # converting the fps to string so that we can display it on frame
        # by using putText function
        fps = str(fps)

        # putting the FPS count on the frame


        start = time.time()
        # boxes = do_detect(m, sized, 0.4, 0.4, False)
        max_prob_obj_cls, output_class_target, overlap_score, boxes = m.detect(torch.tensor(sized), 0, 0)
        finish = time.time()
        print('Predicted in %f seconds.' % (finish - start))

        fps = finish - start

        result_img = plot_boxes_cv2(img, boxes[0], savename=None, class_names=class_names, fps=fps)

        cv2.imshow('Yolo demo', result_img)
        cv2.waitKey(1)

        # if counter % 10 == 0:
        #     cv2.imwrite(f"/Users/razlapid/PhD/MyPapers/Naturalistic Black Box/weights & cfgs/{counter}.png", result_img)
        counter += 1

    cap.release()


def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str, default='/Users/razlapid/PhD/MyPapers/Naturalistic Black Box/weights & cfgs/yolov2.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        default='/Users/razlapid/PhD/MyPapers/Naturalistic Black Box/weights & cfgs/yolov2.weights',
                        help='path of trained model.', dest='weightfile')
    parser.add_argument('-imgfile', type=str,
                        help='path of your image file.', dest='imgfile')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    detect_cv2_camera(args.cfgfile, args.weightfile)
