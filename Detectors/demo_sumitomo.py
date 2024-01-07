import numpy as np
import tensorflow as tf
import os
import torch
from torchvision.transforms import Resize


class detector:
    def __init__(self, conf_theshold=0.3):

        tflite_path = 'Detectors/sumitumo_tflite/ssd_mobilenet_v2_coco_quant_postprocess.tflite'
        absolute_path = os.path.join(
            os.getcwd(), tflite_path)
        self.interpreter = tf.lite.Interpreter(
            model_path=absolute_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.conf_theshold = conf_theshold
        self.name = 'sumitumo_tflite'
        self.epoch_save = 1

    def detect(self, input_imgs, cls_id_attacked=None, epoch=0, with_bbox=False):
        transform = Resize((300, 300))
        new_boxes = []
        all_scores = []
        for i, input_img in enumerate(input_imgs):
            input_img = torch.unsqueeze(input_img, dim=0)
            r_img = transform(input_img)
            r_img = r_img.permute(0, 2, 3, 1)  # (1, 300, 300, 3)
            np_img = r_img.detach().cpu().numpy() * 255.0
            np_img = np_img.astype(np.uint8)
            np_img[0] = np_img[0][:, :, ::-1]

            # set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], np_img)

            # run
            self.interpreter.invoke()
            boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            labels = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
            scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
            if cls_id_attacked is not None:
                people = labels == cls_id_attacked
                boxes = boxes[people]
                scores = scores[people]
            new_boxes.append([])
            for ind, box in enumerate(boxes):
                if scores[ind] > self.conf_theshold:
                    new_boxes[i].append(
                        [box[1], box[0], box[3], box[2], scores[ind], scores[ind], torch.tensor(cls_id_attacked)])
            if len(scores) == 0:
                all_scores.append(0)
            else:
                all_scores.append(max(scores))
        return torch.tensor(all_scores), 0, 0, torch.tensor(new_boxes)

def detect_cv2_camera(cfgfile, weightfile):
    import cv2
    m = DetectorYolov3(cfgfile, weightfile)

    cap = cv2.VideoCapture(0)
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

        start = time.time()
        # boxes = do_detect(m, sized, 0.7, 0.8, False)
        max_prob_obj_cls, output_class_target, overlap_score, boxes = m.detect(torch.tensor(sized), 0, 0)
        finish = time.time()
        print('Predicted in %f seconds.' % (finish - start))

        result_img = plot_boxes_cv2(img, boxes[0], savename=None, class_names=class_names)

        cv2.imshow('Yolo demo', result_img)
        cv2.waitKey(1)

        if counter % 15 == 0:
            cv2.imwrite(f"/Users/razlapid/PhD/Naturalistic-Adversarial-Patch/visualization_imgs/yolo3/img_{counter}.png", result_img)
        counter += 1

    cap.release()

if __name__ == '__main__':
    detect_cv2_camera()