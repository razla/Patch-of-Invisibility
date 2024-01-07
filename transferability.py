import numpy as np
from tqdm import tqdm
from ensemble_tool.utils import *
from ensemble_tool.model import eval_rowPtach, TotalVariation
from PIL import Image
from torchvision.transforms import PILToTensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.utils import save_image
from pprint import pprint
import random

from adversarialYolo.load_data import PatchTransformer, PatchApplier
from new_utils import load_detector, load_dataset

from pathlib import Path
import argparse

def convert_preds_targets(preds, targets, device):
    c_preds = []
    c_targets = []

    for (pred, target) in zip(preds, targets):
        pred_dict = {}
        if pred.nelement() == 0:
            pred_dict['boxes'] = torch.tensor([]).to(device)
            pred_dict['scores'] = torch.tensor([]).to(device)
            pred_dict['labels'] = torch.tensor([]).to(device)
            c_preds.append(pred_dict)
        else:
            # pred_box = pred[0, 0:4].unsqueeze(0).to(device)
            pred_box = target[0,1:].unsqueeze(0).to(device) * 416
            pred_score = pred[0, -2].unsqueeze(0).to(device)
            pred_label = pred[0, -1].unsqueeze(0).to(int).to(device)
            pred_dict['boxes'] = pred_box
            pred_dict['scores'] = pred_score
            pred_dict['labels'] = pred_label
            c_preds.append(pred_dict)

        target_dict = {}
        target_box = target[0,1:].unsqueeze(0).to(device) * 416
        target_label = torch.tensor([0]).to(device)
        target_dict['boxes'] = target_box
        target_dict['labels'] = target_label
        c_targets.append(target_dict)

    return c_preds, c_targets


### -----------------------------------------------------------    Setting     ---------------------------------------------------------------------- ###
Gparser = argparse.ArgumentParser(description='Advpatch Evaluation')
Gparser.add_argument('--seed', default='15089',type=int, help='choose seed')
Gparser.add_argument('--model', default='yolov3', type=str, help='option : yolov2, yolov3, yolov4, yolov5, ssd3, ssd')
Gparser.add_argument('--dataset', default='one_person', type=str, help='option: inria, one_person, test, ron, elbit')
Gparser.add_argument('--tiny', default=False, action='store_true', help='options :True or False')
Gparser.add_argument('--attack', default='fn', type=str, help='options: fn, fp')
Gparser.add_argument('--batch', default=16, type=int, help='batch size')
Gparser.add_argument('--scale', default=0.25, type=float, help='the scale of the patch attached to persons')
Gparser.add_argument('--multi', default=True, action='store_true', help='multi score: true is obj * cls, false is obj')
Gparser.add_argument('--patch', default="/home/razla/NatAdvPatcj/exp/bbgan_yolov3_fn_True_cls_0.1_pop_70_lr_0.02_multi_True_2/generated/generated-images-1000.png", type=str)
apt = Gparser.parse_known_args()[0]
print(apt)

### -----------------------------------------------------------    Setting     ---------------------------------------------------------------------- ###
model_name = apt.model  # options : yolov2, yolov3, yolov4
tiny = apt.tiny  # hint    : only yolov3 and yolov4
dataset_name = apt.dataset  # options : inria, test, ron
attack_num = apt.attack  # options : evasion, FP
patch_scale = apt.scale  # the scale of the patch attached to persons
batch_size = apt.batch  # batch size
patch_path = apt.patch

by_rectangle = True  # True: The patch on the character is "rectangular". / False: The patch on the character is "square"
enable_rotation = False
enable_randomLocation = False
enable_crease = False
enable_projection = False
enable_rectOccluding = False
enable_blurred = False
enable_with_bbox = True  # hint    : It is very time consuming. So, the result is only with bbox at checkpoint.
multi_score = True
epoch_save = 5  # from how many A to save a checkpoint

cls_id_attacked = 0

cls_conf_threshold = 0.5

### ----------------------------------------------------------- Initialization ---------------------------------------------------------------------- ###
# set random seed
Seed = apt.seed  # 37564 7777
torch.manual_seed(Seed)
torch.cuda.manual_seed(Seed)
torch.cuda.manual_seed_all(Seed)
np.random.seed(Seed)
random.seed(Seed)
device = get_default_device()  # cuda or cpu

# confirm folder
global_dir = increment_path(Path('./eval') / f'exp_{attack_num}', exist_ok=False)  # 'checkpoint'
global_dir = Path(global_dir)
checkpoint_dir = global_dir / 'checkpoint'
checkpoint_dir.mkdir(parents=True, exist_ok=True)
sample_dir = global_dir / 'generated'
sample_dir.mkdir(parents=True, exist_ok=True)
print(f"\n##### The results are saved at {global_dir}. #######\n")

### ---------------------------------------------------------- Load generator, detector and dataset -------------------------------------------------------------- ###
detector = load_detector(model_name, tiny, epoch_save, multi_score)
dataset = load_dataset(dataset_name, batch_size, attack_num, train=False)
train_loader = DeviceDataLoader(dataset, device)
total_variation = TotalVariation().to(device)
### ---------------------------------------------------------- Checkpoint & Init -------------------------------------------------------------------- ###
epoch_length_second = len(train_loader)
ep_loss_det = 0
ep_loss_tv = 0
torch.cuda.empty_cache()
writer = init_tensorboard(path=global_dir, name="gan_adversarial")
### ---------------------------------------------------------- Loading Patch -------------------------------------------------------------------- ###
patch = Image.open(patch_path)
patch = PILToTensor()(patch).cuda() / 255.0

# init and show the length of one epoch
print(f'One epoch length is {len(train_loader)}')
patch_transformer = PatchTransformer().to(device)
patch_applier = PatchApplier().to(device)
p_img_batch = []

metric = MeanAveragePrecision()

total_loss_det = np.zeros(1)
### -----------------------------------------------------------    Training    ---------------------------------------------------------------------- ###
for epoch in range(1, 2):
    ep_loss_det = 0
    for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'2 Running epoch {epoch}',
                                                total=epoch_length_second):
        with torch.autograd.set_detect_anomaly(True):
            loss_det, p_img_batch, bboxes = eval_rowPtach(img_batch, lab_batch, patch_scale, cls_id_attacked
                , model_name, detector
                , patch_transformer, patch_applier
                , by_rectangle
                , enable_rotation
                , enable_randomLocation
                , enable_crease
                , enable_projection
                , enable_rectOccluding
                , enable_blurred
                , enable_with_bbox
                , cls_conf_threshold
                , patch_mode = 0
                , enable_no_random=True
                , fake_images_default=patch)
            # # Record loss and score
            preds, targets = convert_preds_targets(bboxes, lab_batch, device)
            metric.update(preds, targets)
            ep_loss_det += loss_det

    ep_loss_det = ep_loss_det / epoch_length_second

print(ep_loss_det)
pprint(metric.compute())