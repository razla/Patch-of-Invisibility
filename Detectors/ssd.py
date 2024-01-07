from torchvision.models.detection import SSD300_VGG16_Weights
import torchvision
import torch

class SSD:
    def __init__(self, show_detail=False, tiny=False, epoch_save=10, multi_score=True, conf_theshold=0.3):
        self.model = torchvision.models.detection.ssd300_vgg16(
            weights=SSD300_VGG16_Weights.DEFAULT)
        self.model.eval().cuda()
        self.conf_theshold = conf_theshold
        self.epoch_save = epoch_save
        self.multi_score = multi_score
        self.name = 'ssd3'
        self.device = torch.device("cuda")
        self.epoch_save = epoch_save

    def detect(self, input_imgs, cls_id_attacked=None, epoch=0, clear_imgs=None, with_bbox=False):
        preds = self.model(input_imgs)
        new_boxes = []
        all_scores = []
        for i, pred in enumerate(preds):
            if cls_id_attacked is not None:
                people = pred['labels'] == cls_id_attacked + 1
                boxes = pred['boxes'][people]
                scores = pred['scores'][people]
            else:
                boxes = pred['boxes'] / input_imgs.shape[-1]
                scores = pred['scores']
                cls_id_attacked = -1

            new_boxes.append([])
            for ind, box in enumerate(boxes):
                if scores[ind] > self.conf_theshold:
                    new_boxes[i].append(
                        [box[1], box[0], box[3], box[2], scores[ind], scores[ind], torch.tensor(cls_id_attacked + 1)])
            if len(scores) == 0:
                all_scores.append(torch.tensor(0.0, requires_grad=True).to(self.device))
            else:
                all_scores.append(torch.max(scores))
        return torch.stack(all_scores), 0, 0, new_boxes