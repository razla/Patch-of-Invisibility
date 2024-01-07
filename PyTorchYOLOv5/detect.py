import sys
sys.path.insert(0, "/home/eylonmiz/NatadvPatch/PyTorchYOLOv5/yolov5")
from models.common import DetectMultiBackend, AutoShape

import numpy as np
import torch.nn as nn
import torch
from shapely.geometry import Polygon, box


class DetectorYolov5:
    def __init__(self, tiny=True, multi_score=True, conf_theshold=0.25):
        self.device = torch.device("cuda")
        # weights_path = "/home/eylonmiz/NatadvPatch/PyTorchYOLOv5/weights/yolov5n.pt"
        weights_path = "/home/eylonmiz/NatadvPatch/PyTorchYOLOv5/weights/yolov5s.pt" if tiny else \
            "/home/eylonmiz/NatadvPatch/PyTorchYOLOv5/weights/yolov5m.pt"
        self.model = AutoShape(DetectMultiBackend(weights_path, self.device))
        self.model.eval()
        self.conf_thesh = conf_theshold
        self.multi_score = multi_score

    def detect(self, batch_images, batch_labels=None, cls_id_attacked=0, is_train=True):
        """NOTE: INPUTS:
        batch_images.shape = (batch_size, 3, 416, 416)
        batch_labels.shape = (batch_size, MAX_NUM_LABELS, cls_id_dim+normalized_xyxy_coords_dim) = (batch_size, 14, 5)
        Each instance in batch_labels has shape of (14, 5).
        Each of these rows will look like [1., 1., 1., 1., 1.] if its redundant (empty entries)
        cls_id_attacked: int 
        
        NOTE: OUTPUTS:
        max_prob_obj_cls.shape = (batch_size,) 
        len(bbox) = batch_size, bbox[0].shape = (n_boxes, len([*xyxy_coords, conf, conf+obj, cls_id])) = (n_boxes, 4+1+1+1)
        output_class_target is nonsense"""
        
        # NOTE: Do we need to also use F.interpolate?
        # * 255 because the AutoShape YOLOv5 wrapper expects a uint8 input images, 
        # but the input images were passed in ToTensor transform, which divides the images by 255"""
        img_size = batch_images[0].shape[1]
        batch_images = [(255 * img.cpu().permute(1, 2, 0).numpy()).astype(np.uint8) for img in batch_images]
        
        batch_preds = self.model(batch_images).xyxy
        new_batch_preds = []
        scores = []
        
        new_batch_labels = None
        if is_train:
            # Preserving only the non-empty entries and de-normalizing labels bboxes
            len_label_row = 5
            batch_labels = batch_labels[:, :, 1:len_label_row]
            new_batch_labels = list()
            for sample_labels in batch_labels:
                sample_labels = sample_labels[sample_labels != torch.ones((sample_labels.shape[-1],), device=self.device)]
                sample_labels = torch.reshape(sample_labels, (len(sample_labels)//(len_label_row-1), (len_label_row-1)))
                sample_labels *= img_size  # De-normalizing
                new_batch_labels.append(sample_labels)
        
        for sample_idx in range(len(batch_preds)):
            sample_preds = batch_preds[sample_idx]
            
            # Filter a single image detections by the desired class_id and conf_thesh
            sample_preds = sample_preds[torch.where(
                (sample_preds[:, 6] == torch.tensor(cls_id_attacked, device=self.device)) & \
                (sample_preds[:, 5] >= torch.tensor(self.conf_thesh, device=self.device)))]
            
            if not is_train:
                if sample_preds.numel() == 0:
                    scores.append(torch.tensor([0.0 for _ in range(batch_labels.shape[1])], device=sample_preds.device))
                    # new_batch_preds.append([torch.empty((0, 7), device=sample_preds.device) for _ in range(batch_labels.shape[1])])
                else:
                    scores.append(torch.tensor([p[5 if self.multi_score else 4] for p in sample_preds], device=sample_preds[0].device))
                    new_batch_preds.append(sample_preds)
            else:
                # For each sample image (from the batch) - pick the pred with the best IoU with some label
                sample_labels = new_batch_labels[sample_idx]
                best_iou = 0.0
                best_pred_idx = None
                for sample_pred_idx, sample_pred in enumerate(sample_preds):
                    sample_pred_bbox = box(*sample_pred[:4].cpu().tolist())
                    for label in sample_labels:
                        sample_label_bbox = box(*label.cpu().tolist())
                        iou = sample_pred_bbox.intersection(sample_label_bbox).area / sample_pred_bbox.union(sample_label_bbox).area
                        if iou > best_iou:
                            best_iou = iou
                            best_pred_idx = sample_pred_idx
                            
                if len(sample_preds) == 0 or best_pred_idx is None:
                    scores.append(torch.empty((0,)).to(self.device))
                    new_batch_preds.append(torch.empty((0, 7)).to(self.device))
                else:
                    best_sample_pred = sample_preds[best_pred_idx]
                    scores.append(best_sample_pred[5 if self.multi_score else 4])
                    new_batch_preds.append([best_sample_pred])
                    
        return scores, new_batch_preds
    