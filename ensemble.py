import numpy as np
from copy import deepcopy
from tqdm import tqdm
from ensemble_tool import model
from ensemble_tool.utils import *
from ensemble_tool.model import train_rowPtach, TotalVariation, TotalVariation2
import random

from adversarialYolo.load_data import PatchTransformer, PatchApplier
from new_utils import load_generator
from new_utils import load_classifier
from new_utils import load_optimizer
from new_utils import load_detector
from new_utils import load_dataset
from new_utils import create_plots

from GANLatentDiscovery.utils import is_conditional
from square_attack import init_square_attack_strips_patch_l_inf

from pathlib import Path
import argparse

n = 50
lr = 1e-2
min_lr = 1e-2
max_lr = 5e-5

# TODO:
# CHECK RANDOM GAN PATCH PATCHES SAVING


### -----------------------------------------------------------    Setting     ---------------------------------------------------------------------- ###
Gparser = argparse.ArgumentParser(description='Advpatch Training')
Gparser.add_argument('--seed', default='15089',type=int, help='choose seed')
Gparser.add_argument('--model', default='yolov4', type=str, help='option : yolov2, yolov3, yolov4, yolov5, ssd3, ssd')
Gparser.add_argument('--dataset', default='one_person', type=str, help='option: inria, one_person, test, ron, elbit')
Gparser.add_argument('--tiny', default=True, action='store_true', help='options :True or False')
Gparser.add_argument('--method', default='bbgan', type=str, help='options: raw, random_raw, random_gan, bbgan, wbgan, nes, nes_est, square')
Gparser.add_argument('--attack', default='fn', type=str, help='options: fn, fp, dos')
Gparser.add_argument('--epochs', default=1000, type=int, help='num of epochs')
Gparser.add_argument('--batch', default=8, type=int, help='batch size')
Gparser.add_argument('--lr', default=0.02, type=float, help='learning rate')
Gparser.add_argument('--opt', default='adam', type=str, help='optimizer: adam, lion')
Gparser.add_argument('--scale', default=0.25, type=float, help='the scale of the patch attached to persons')
Gparser.add_argument('--multi', default=True, action='store_true', help='multi score: true is obj * cls, false is obj')
Gparser.add_argument('--weightTV', default=0.1, type=float, help='the weight of the tv loss')
Gparser.add_argument('--weightCLS', default=0.1, type=float, help='the weight of the pred loss (for FP attack)')
Gparser.add_argument('--classGAN', default=259, type=int, help='class in big gan') # 84:peacock
Gparser.add_argument('--classDET', default=0, type=int, help='The attacked class (0: person)')
Gparser.add_argument('--classTGT', default=1, type=int, help='For the evasion attack - the target class')
Gparser.add_argument('--pop', default=50, type=int, help="size of population")
Gparser.add_argument('--sigma', default=0.1, type=float, help="sigma")
apt = Gparser.parse_known_args()[0]
print(apt)

### -----------------------------------------------------------    Setting     ---------------------------------------------------------------------- ###
model_name            = apt.model       # options : yolov2, yolov3, yolov4, yolov5
tiny                  = apt.tiny        # hint    : only yolov3 and yolov4, yolov5
dataset_name          = apt.dataset     # options : inria, test, ron
method_name            = apt.method      # options : raw patch, BigGAN
attack_name            = apt.attack      # options : evasion, FP
patch_scale           = apt.scale       # the scale of the patch attached to persons
n_epochs              = apt.epochs      # training total epoch
batch_size            = apt.batch       # batch size
lr                    = apt.lr          # learning rate. (hint v3~v4(~0.02) v2(~0.01))
opt_name              = apt.opt         # optimizer
multi_score           = apt.multi       # multiscore
cls_id_attacked       = apt.classDET    # the class attacked. (0: person). List: https://gist.github.com/AruniRC/7b3dadd004da04c80198557db5da4bda
cls_id_generation     = apt.classGAN    # the class generated at patch. (259: pomeranian) List: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
cls_id_tgt            = apt.classTGT    # the target class for evasion attack
weight_loss_tv        = apt.weightTV    # total variation loss rate    ([0-0.1])
weight_loss_cls       = apt.weightCLS    # cls loss weight (for FP attack)
n_pop                 = apt.pop
sigma                 = apt.sigma
by_rectangle          = False            # True: The patch on the character is "rectangular". / False: The patch on the character is "square"
enable_rotation       = True
enable_randomLocation = False
enable_crease         = False
enable_projection     = False
enable_rectOccluding  = False
enable_blurred        = False
enable_with_bbox      = True            # hint    : It is very time consuming. So, the result is only with bbox at checkpoint.
epoch_save            = 10               # from how many A to save a checkpoint
rawPatch_size         = 128             # the size of patch without gan. It's just like "https://openaccess.thecvf.com/content_CVPRW_2019/html/CV-COPS/Thys_Fooling_Automated_Surveillance_Cameras_Adversarial_Patches_to_Attack_Person_Detection_CVPRW_2019_paper.html"
max_value_latent_item = 20              # the max value of latent vectors
### ----------------------------------------------------------- Initialization ---------------------------------------------------------------------- ###
# set random seed
Seed = apt.seed
torch.manual_seed(Seed)
torch.cuda.manual_seed(Seed)
torch.cuda.manual_seed_all(Seed)
np.random.seed(Seed)
random.seed(Seed)
device = get_default_device() # cuda or cpu

# confirm folder
experiments_dir = Path("/home/eylonmiz/NatadvPatch/exp")
global_dir = increment_path(experiments_dir / f'seed-{Seed}_method-{method_name}_model-{model_name}_attak-{attack_name}_tiny-{tiny}_clsw-{weight_loss_cls}_pop-{n_pop}_lr-{lr}_sigma-{sigma}_multi-{multi_score}', exist_ok=False)
global_dir = Path(global_dir)
checkpoint_dir = global_dir / 'checkpoint'
checkpoint_dir.mkdir(parents=True, exist_ok=True)
sample_dir = global_dir / 'generated'
sample_dir.mkdir(parents=True, exist_ok=True)
print(f"\n##### The results are saved at {global_dir}. #######\n")

### ---------------------------------------------------------- Load generator, detector and dataset -------------------------------------------------------------- ###
generator, len_z, len_latent = load_generator(method_name)
detector = load_detector(model_name, tiny, epoch_save, multi_score)
classifier, classifier_weights = load_classifier('resnet50', device)
dataset = load_dataset(dataset_name, batch_size, attack_name)
train_loader = DeviceDataLoader(dataset, device)
total_variation = TotalVariation().to(device)
if method_name == 'nes':
    total_variation = TotalVariation2().to(device)
if is_conditional(generator):
    generator.set_classes(cls_id_generation)
if attack_name == 'fp':
    detector.max_prob_extractor.set_cls_id_tgt(cls_id_tgt)
if attack_name == 'xai':
    target_xai = torch.zeros((416, 416)).to(device)
    target_xai[0:100, 0:100] = 1
else:
    target_xai = None
### ---------------------------------------------------------- Checkpoint & Init -------------------------------------------------------------------- ###
epoch_length_second  = len(train_loader)
ep_loss_det   = 0
ep_loss_tv    = 0
torch.cuda.empty_cache()
if method_name == 'raw':
    rawPatch = torch.rand((3, rawPatch_size, rawPatch_size), device=device).requires_grad_(True)
    opt = load_optimizer(opt_name, rawPatch, lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=50)
elif method_name == 'random_raw':
    rawPatch = torch.randn((3, rawPatch_size, rawPatch_size), device=device).requires_grad_(False)
    latent_vector = rawPatch
    opt = None
    scheduler = None
elif method_name == 'square':
    rawPatch = init_square_attack_strips_patch_l_inf((rawPatch_size, rawPatch_size)).to(device).requires_grad_(False)
    latent_vector = rawPatch
    opt = None
    scheduler = None
elif method_name == 'random_gan':
    latent_vector = torch.normal(0.0, torch.ones(len_latent)).to(device).requires_grad_(False)
    opt = None
    scheduler = None
elif method_name == 'nes_est':
    rawPatch = torch.randn((3, rawPatch_size, rawPatch_size), device=device).requires_grad_(True)  # NOTE: Does this need to be requires_grad_(True)?
    latent_vector = rawPatch
    opt = load_optimizer(opt_name, latent_vector, lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=50)
else:
    latent_vector = torch.normal(0.0, torch.ones(len_latent)).to(device).requires_grad_(True)
    opt = load_optimizer(opt_name, latent_vector, lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=50)

# init and show the length of one epoch
print(f'One epoch length is {len(train_loader)}')
patch_transformer = PatchTransformer().to(device)
patch_applier = PatchApplier().to(device)
p_img_batch = []

total_loss_det = np.zeros(n_epochs)
total_loss_tv  = np.zeros(n_epochs)
total_loss_cls = np.zeros(n_epochs)
total_loss_xai = np.zeros(n_epochs)
total_loss     = np.zeros(n_epochs)
### -----------------------------------------------------------    Training    ---------------------------------------------------------------------- ###
for epoch in range(n_epochs):
    ep_loss_det = 0
    ep_loss_tv  = 0
    ep_loss_cls = 0
    ep_loss_xai = 0
    ep_loss_total = 0
    last_latent = deepcopy(latent_vector)
    for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'2 Running epoch {epoch}',total=epoch_length_second):
        with torch.autograd.set_detect_anomaly(True):
            if method_name == 'raw':
                rawPatch.data = torch.round(rawPatch.data * 10000) * (10 ** -4)
                latent_vector = rawPatch
            else:
                latent_vector.data = torch.round(latent_vector.data * 10000) * (10**-4)
                latent_vector.data = torch.clamp(latent_vector.data, -max_value_latent_item, max_value_latent_item)
            
            loss_det, loss_tv, loss_cls, loss_xai, loss_total, p_img_batch = train_rowPtach(method_name=method_name, epoch=epoch, generator=generator,
                                                                          opt=opt, device=device,
                                                                          latent_vector=latent_vector,
                                                                          input_imgs=img_batch, label=lab_batch,
                                                                          patch_scale=patch_scale, cls_id_attacked=cls_id_attacked,
                                                                          model_name=model_name, detector=detector,
                                                                          patch_transformer=patch_transformer, patch_applier=patch_applier,
                                                                          total_variation=total_variation, by_rectangle=by_rectangle,
                                                                          enable_rotation=enable_rotation, enable_randomLocation=enable_randomLocation,
                                                                          enable_crease=enable_crease, enable_projection=enable_projection,
                                                                          enable_rectOccluding=enable_rectOccluding, enable_blurred=enable_blurred,
                                                                          enable_with_bbox=enable_with_bbox,
                                                                            weight_loss_tv=weight_loss_tv, weight_loss_cls=weight_loss_cls, attack_name=attack_name,
                                                                            classifier=classifier, classifier_weights=classifier_weights, n_pop=n_pop, sigma=sigma, target_xai=target_xai)
            # # Record loss and score
            ep_loss_det   += loss_det
            ep_loss_tv    += loss_tv
            ep_loss_cls   += loss_cls
            ep_loss_xai   += loss_xai
            ep_loss_total += loss_total
            
            if method_name == 'random_raw' or method_name == 'random_gan' or method_name == 'square':
                latent_vector = model.best_latent

    ep_loss_det   = ep_loss_det/epoch_length_second
    ep_loss_tv    = ep_loss_tv/epoch_length_second
    ep_loss_cls   = ep_loss_cls/epoch_length_second
    ep_loss_xai   = ep_loss_xai/epoch_length_second
    ep_loss_total = ep_loss_total/epoch_length_second

    ep_loss = ep_loss_det

    if method_name != 'random_raw' and method_name != 'random_gan' and method_name != 'square':
        scheduler.step(ep_loss)

    ep_loss_det      = ep_loss_det.detach().cpu().numpy()
    ep_loss_tv       = ep_loss_tv.detach().cpu().numpy()
    ep_loss_cls      = ep_loss_cls.detach().cpu().numpy()
    ep_loss_xai      = ep_loss_xai.detach().cpu().numpy()
    ep_loss_total    = ep_loss_total.detach().cpu().numpy()

    total_loss_det[epoch] = ep_loss_det
    total_loss_tv[epoch] = ep_loss_tv
    total_loss_cls[epoch] = ep_loss_cls
    total_loss_xai[epoch] = ep_loss_xai
    total_loss[epoch] = ep_loss_total
        
    print("ep_loss_det      : "+str(ep_loss_det))
    print("ep_loss_tv       : "+str(ep_loss_tv))
    print("ep_loss_cls      : "+str(ep_loss_cls))
    print("ep_loss_xai      : "+str(ep_loss_xai))
    print("ep_loss_total    : "+str(ep_loss_total))
    print("latent code:     :'"+f"norn_inf:{torch.max(torch.abs(latent_vector)):.4f}; norm_1:{torch.norm(latent_vector, p=1)/latent_vector.shape[0]:.4f}")
    print("current-last latent l2 distance: "+f"{torch.norm(latent_vector-last_latent, p=2):.4f}")

    if method_name == 'raw' or method_name == 'random_raw' or method_name == 'nes_est' or method_name == 'square':
        # save patch
        rawPatch = latent_vector
        save_samples(index=epoch, sample_dir=sample_dir, patch=rawPatch.cpu())
    else:
        # save patch
        print(f"Save at: {global_dir}")
        save_samples_GANLatentDiscovery(index=epoch, sample_dir=sample_dir, G=generator,
                                        latent_vector=latent_vector,
                                        device=device)

    if epoch % epoch_save == 0:
        # save the patched image
        save_the_patched(index=epoch, the_patched=p_img_batch, sample_dir=sample_dir, show=False)
        # if method_name != "random_raw" and method_name != "random_gan" and method_name != "nes_est":
        PATH   = str(checkpoint_dir) + "/params"+str(epoch)+".pt"
        d = {'epoch': epoch,
            'latent_vector':latent_vector.data,
            'ep_loss_total':ep_loss_total}
        if method_name == "bbgan" or method_name == "nes_est" or method_name == "random_gan":
            d.update({
                'ep_loss_tv': ep_loss_tv,
                'ep_loss_det': ep_loss_tv,
                'ep_loss_cls': ep_loss_cls})
            if method_name == "bbgan" or method_name == "nes_est":
                d.update({'optimizer_state_dict': opt.state_dict(), 'scheduler_state_dict': scheduler.state_dict()})
        torch.save(d, PATH)
        print(f"save checkpoint: "+str(PATH))
create_plots(total_loss_det, total_loss_tv, total_loss_cls, total_loss, global_dir)

print('Total loss det:')
print(total_loss_det)
print('Total loss tv:')
print(total_loss_tv)
print('Total loss cls:')
print(total_loss_cls)
print('Total loss:')
print(total_loss)