import torch
import torch.nn as nn
import numpy as np
from pytorchYOLOv4.tool.utils import load_class_names
import torchvision
from torchvision import transforms
# from pytorch_grad_cam import EigenCAM, GradCAM
from torchvision.utils import save_image
import torch.nn.functional as F
from PIL import ImageDraw, ImageFont
from copy import deepcopy

# import _saved_tensors

from GANLatentDiscovery.visualization import evo_interpolate_shift, interpolate_shift
from new_utils import classifier_predict, show_images_with_bbox
from square_attack import update_square_attack_patch_l_inf


best_loss = np.inf
best_latent = None
best_p_img_batch = None

class TotalVariation(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total vatiation (TV) of an patch.

    """

    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, adv_patch):
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1]+0.000001),0)
        tvcomp1 = torch.sum(torch.sum(tvcomp1,0),0)
        tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :]+0.000001),0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2,0),0)
        tv = tvcomp1 + tvcomp2
        return tv/torch.numel(adv_patch)

class TotalVariation2(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total vatiation (TV) of an patch.

    """

    def __init__(self):
        super(TotalVariation2, self).__init__()

    def forward(self, adv_patch):
        if len(adv_patch.shape) == 3:
            tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1]+0.000001),0)
            tvcomp1 = torch.sum(torch.sum(tvcomp1,0),0)
            tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :]+0.000001),0)
            tvcomp2 = torch.sum(torch.sum(tvcomp2,0),0)
            tv = tvcomp1 + tvcomp2
            return tv/torch.numel(adv_patch)
        else:
            tv_h = torch.pow(adv_patch[:, :, 1:, :] - adv_patch[:, :, :-1, :], 2).sum(dim=[1, 2, 3])
            tv_w = torch.pow(adv_patch[:, :, :, 1:] - adv_patch[:, :, :, :-1], 2).sum(dim=[1, 2, 3])
            tv = tv_h + tv_w
            return tv / torch.numel(adv_patch[0])

def eval_rowPtach(input_imgs, label, patch_scale, cls_id_attacked
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
                , fake_images_default=None
                  , clean=False):

    fake_images = fake_images_default

    # enable_empty_patch == False: no patch
    enable_empty_patch = False
    if(patch_mode == 1):
        enable_empty_patch = True
    p_img_batch = input_imgs
    with torch.no_grad():
        if not clean:
            adv_batch_t, adv_patch_set, msk_batch = patch_transformer(  adv_patch=fake_images,
                                                                        lab_batch=label,
                                                                        img_size=416,
                                                                        patch_mask=[],
                                                                        by_rectangle=by_rectangle,
                                                                        do_rotate=enable_rotation,
                                                                        rand_loc=enable_randomLocation,
                                                                        scale_rate = patch_scale,
                                                                        with_crease=enable_crease,
                                                                        with_projection=enable_projection,
                                                                        with_rectOccluding=enable_rectOccluding,
                                                                        enable_empty_patch=enable_empty_patch,
                                                                        enable_no_random=enable_no_random,
                                                                        enable_blurred=enable_blurred)
            p_img_batch = patch_applier(input_imgs, adv_batch_t)

        max_prob_obj_cls, output_class_target, bboxes = None, None, None
        if model_name == "yolov3" or model_name == "yolov2" or model_name == "yolov4":
            max_prob_obj_cls, output_class_target, bboxes = detector.detect(input_imgs=p_img_batch,
                                                                      cls_id_attacked=cls_id_attacked, epoch=0, with_bbox=enable_with_bbox)
        elif model_name == "yolov5":
            max_prob_obj_cls, bboxes = detector.detect(p_img_batch, label, cls_id_attacked)
            max_prob_obj_cls = torch.tensor([p for p in max_prob_obj_cls if p.numel() > 0], device=max_prob_obj_cls[0].device)
            max_prob_obj_cls = torch.tensor(0.0) if len(max_prob_obj_cls) == 0 else max_prob_obj_cls
            
        max_prob_obj_cls, output_class_target, bboxes = detector.detect(input_imgs=p_img_batch, cls_id_attacked=cls_id_attacked, epoch=0, with_bbox=enable_with_bbox)
        loss_det = torch.mean(max_prob_obj_cls)

        # darw bbox
        if enable_with_bbox and len(bboxes)>0:
            trans_2pilimage = transforms.ToPILImage()
            batch = p_img_batch.size()[0]
            for b in range(batch):
                img_pil = trans_2pilimage(p_img_batch[b].cpu())
                img_width = img_pil.size[0]
                img_height = img_pil.size[1]
                namesfile = 'pytorchYOLOv4/data/coco.names'
                class_names = load_class_names(namesfile)
                # sample first image
                # print("bbox : "+str(bbox))
                bbox = bboxes[b]
                for box in bbox:
                    # print("box size : "+str(box.size()))
                    cls_id = box[6].int()
                    cls_name = class_names[cls_id]
                    cls_conf = box[5]
                    if(cls_id == cls_id_attacked):
                        if(cls_conf > cls_conf_threshold):
                            if(model_name == "yolov2"):
                                x_center    = box[0]
                                y_center    = box[1]
                                width       = box[2]
                                height      = box[3]
                                left        = (x_center.item() - width.item() / 2) * img_width
                                right       = (x_center.item() + width.item() / 2) * img_width
                                top         = (y_center.item() - height.item() / 2) * img_height
                                bottom      = (y_center.item() + height.item() / 2) * img_height
                            if(model_name == "yolov4") or (model_name == "yolov3"):
                                left        = int(box[0] * img_width)
                                right       = int(box[2] * img_width)
                                top         = int(box[1] * img_height)
                                bottom      = int(box[3] * img_height)
                            # img with prediction
                            draw = ImageDraw.Draw(img_pil)
                            shape = [left, top, right, bottom]
                            draw.rectangle(shape, outline ="red")
                            # text
                            color = [255,0,0]
                            font = ImageFont.truetype("cmb10.ttf", int(min(img_width, img_height)/18))
                            sentence = str(cls_name)+" ("+str(round(float(cls_conf), 2))+")"
                            position = [left, top]
                            draw.text(tuple(position), sentence, tuple(color), font=font)
                trans_2tensor = transforms.ToTensor()
                p_img_batch[b] = trans_2tensor(img_pil)

    return loss_det, p_img_batch, bboxes

def train_rowPtach(method_name, epoch, generator
                    , opt, device
                    , latent_vector
                    , input_imgs, label, patch_scale, cls_id_attacked
                    , model_name, detector
                    , patch_transformer, patch_applier
                    , total_variation
                    , by_rectangle
                    , enable_rotation
                    , enable_randomLocation
                    , enable_crease
                    , enable_projection
                    , enable_rectOccluding
                    , enable_blurred
                    , enable_with_bbox
                    , weight_loss_tv
                    , weight_loss_cls
                    , attack_name
                    , classifier = None
                    , classifier_weights = None
                    , n_pop = None
                    , sigma = None
                    , target_xai = None
                    ):
    # raw patch with grad
    if method_name == 'raw':
        return train_raw_patch(epoch, opt,
                        latent_vector, input_imgs,
                        label, patch_scale,
                        cls_id_attacked, model_name,
                        detector, patch_transformer,
                        patch_applier, total_variation,
                        by_rectangle, enable_rotation,
                        enable_randomLocation, enable_crease,
                        enable_projection, enable_rectOccluding,
                        enable_blurred, enable_with_bbox, weight_loss_tv, weight_loss_cls, attack_name)
    if method_name == 'random_raw':
        return train_random_patch(epoch, opt,
                        latent_vector, input_imgs,
                        label, patch_scale,
                        cls_id_attacked, model_name,
                        detector, patch_transformer,
                        patch_applier, total_variation,
                        by_rectangle, enable_rotation,
                        enable_randomLocation, enable_crease,
                        enable_projection, enable_rectOccluding,
                        enable_blurred, enable_with_bbox, weight_loss_tv, weight_loss_cls, attack_name)
    if method_name == 'square':
        return train_square_attack(epoch, opt,
                        latent_vector, input_imgs,
                        label, patch_scale,
                        cls_id_attacked, model_name,
                        detector, patch_transformer,
                        patch_applier, total_variation,
                        by_rectangle, enable_rotation,
                        enable_randomLocation, enable_crease,
                        enable_projection, enable_rectOccluding,
                        enable_blurred, enable_with_bbox, weight_loss_tv, weight_loss_cls, attack_name)
    if method_name == 'random_gan':
        return train_random_patch_gan(epoch, generator
                    , opt, device
                    , latent_vector
                    , input_imgs, label, patch_scale, cls_id_attacked
                    , model_name, detector
                    , patch_transformer, patch_applier
                    , total_variation
                    , by_rectangle
                    , enable_rotation
                    , enable_randomLocation
                    , enable_crease
                    , enable_projection
                    , enable_rectOccluding
                    , enable_blurred
                    , enable_with_bbox
                    , weight_loss_tv
                    , weight_loss_cls
                    ,attack_name
                    , classifier
                    , classifier_weights
                    , n_pop
                    , sigma
                   )
    # gan patch without grad
    if method_name == 'bbgan':
        return evo_train_gan_patch(epoch, generator
                    , opt, device
                    , latent_vector
                    , input_imgs, label, patch_scale, cls_id_attacked
                    , model_name, detector
                    , patch_transformer, patch_applier
                    , total_variation
                    , by_rectangle
                    , enable_rotation
                    , enable_randomLocation
                    , enable_crease
                    , enable_projection
                    , enable_rectOccluding
                    , enable_blurred
                    , enable_with_bbox
                    , weight_loss_tv
                    , weight_loss_cls
                    , attack_name
                    , classifier
                    , classifier_weights
                    , n_pop
                    , sigma)
    if method_name == 'nes_est':
        return nes_est_train_patch(epoch, generator
                    , opt, device
                    , latent_vector
                    , input_imgs, label, patch_scale, cls_id_attacked
                    , model_name, detector
                    , patch_transformer, patch_applier
                    , total_variation
                    , by_rectangle
                    , enable_rotation
                    , enable_randomLocation
                    , enable_crease
                    , enable_projection
                    , enable_rectOccluding
                    , enable_blurred
                    , enable_with_bbox
                    , weight_loss_tv
                    , weight_loss_cls
                    , attack_name
                    , classifier
                    , classifier_weights
                    , n_pop
                    , sigma)
    if method_name == 'nes':
        return nes_train_patch(epoch, generator
                    , opt, device
                    , latent_vector
                    , input_imgs, label, patch_scale, cls_id_attacked
                    , model_name, detector
                    , patch_transformer, patch_applier
                    , total_variation
                    , by_rectangle
                    , enable_rotation
                    , enable_randomLocation
                    , enable_crease
                    , enable_projection
                    , enable_rectOccluding
                    , enable_blurred
                    , enable_with_bbox
                    , weight_loss_tv
                    , weight_loss_cls
                    , attack_name
                    , classifier
                    , classifier_weights
                    , n_pop
                    , sigma
                    , target_xai)
    # gan patch with grad
    if method_name == 'wbgan':
        return train_gan_patch(epoch, generator
                    , opt, device
                    , latent_vector
                    , input_imgs, label, patch_scale, cls_id_attacked
                    , model_name, detector
                    , patch_transformer, patch_applier
                    , total_variation
                    , by_rectangle
                    , enable_rotation
                    , enable_randomLocation
                    , enable_crease
                    , enable_projection
                    , enable_rectOccluding
                    , enable_blurred
                    , enable_with_bbox
                    , weight_loss_tv
                    , weight_loss_cls
                    ,attack_name
                    , classifier
                    , classifier_weights
                    , target_xai
                   )

def train_raw_patch(epoch, opt,
                        latent_vector, input_imgs,
                        label, patch_scale,
                        cls_id_attacked, model_name,
                        detector, patch_transformer,
                        patch_applier, total_variation,
                        by_rectangle, enable_rotation,
                        enable_randomLocation, enable_crease,
                        enable_projection, enable_rectOccluding,
                        enable_blurred, enable_with_bbox, weight_loss_tv, weight_loss_cls, attack_name):
    opt.zero_grad()
    adv_batch_t, adv_patch_set, msk_batch = patch_transformer(adv_patch=latent_vector,
                                                              lab_batch=label,
                                                              img_size=416,
                                                              patch_mask=[],
                                                              by_rectangle=by_rectangle,
                                                              do_rotate=enable_rotation,
                                                              rand_loc=enable_randomLocation,
                                                              scale_rate=patch_scale,
                                                              with_crease=enable_crease,
                                                              with_projection=enable_projection,
                                                              with_rectOccluding=enable_rectOccluding,
                                                              enable_blurred=enable_blurred)
    p_img_batch = patch_applier(input_imgs, adv_batch_t)

    max_prob_obj_cls, output_class_target, bboxes = None, None, None
    if model_name == "yolov3" or model_name == "yolov2" or model_name == "yolov4":
        max_prob_obj_cls, output_class_target, bboxes = detector.detect(input_imgs=p_img_batch,
                                                                    cls_id_attacked=cls_id_attacked, epoch=epoch, with_bbox=enable_with_bbox)
    elif model_name == "yolov5":
        max_prob_obj_cls, bboxes = detector.detect(p_img_batch, label, cls_id_attacked)
        max_prob_obj_cls = torch.tensor([p for p in max_prob_obj_cls if p.numel() > 0], device=max_prob_obj_cls[0].device)
        max_prob_obj_cls = torch.tensor(0.0, device=latent_vector.device) if len(max_prob_obj_cls) == 0 else max_prob_obj_cls
            
        det_loss = torch.mean(max_prob_obj_cls)

    if attack_name == 'fn':
        tv_loss = total_variation(latent_vector)
        total_loss = det_loss + (weight_loss_tv * tv_loss)
    elif attack_name == 'fp':
        tv_loss = total_variation(latent_vector)
        det_loss = (1 - det_loss)
        total_loss = det_loss + (weight_loss_tv * tv_loss)
    # elif attack_name == 'dos':
    #     tv_loss = total_variation(latent_vector)
    #     max_boj = _saved_tensors.nms_scores
    #     areas = _saved_tensors.nms_boxes[:, :, 2].reshape(-1) * _saved_tensors.nms_boxes[:, :, 3].reshape(-1)
    #     bbox_area = torch.mean(areas)
    #     det_loss = bbox_area * 50 + max_boj * 500
    #     total_loss = bbox_area * 50 + max_boj * 500 + (weight_loss_tv * tv_loss)
    total_loss.backward()
    opt.step()

    # draw bbox at output
    if enable_with_bbox and len(bboxes) > 0 and (epoch % detector.epoch_save == 0):
        p_img_batch = show_images_with_bbox(p_img_batch, bboxes, cls_id_attacked, model_name)

    return det_loss, tv_loss, torch.tensor(0), torch.tensor(0), total_loss, p_img_batch

def train_random_patch(epoch, opt,
                        latent_vector, input_imgs,
                        label, patch_scale,
                        cls_id_attacked, model_name,
                        detector, patch_transformer,
                        patch_applier, total_variation,
                        by_rectangle, enable_rotation,
                        enable_randomLocation, enable_crease,
                        enable_projection, enable_rectOccluding,
                        enable_blurred, enable_with_bbox, weight_loss_tv, weight_loss_cls,
                        attack_name, classifier=None, classifier_weights=None):
    global best_loss, best_latent, best_p_img_batch
    with torch.no_grad():
        
        perturbed_latent_vector = latent_vector + torch.randn(latent_vector.shape).cuda() * 0.001
        
        adv_batch_t, adv_patch_set, msk_batch = patch_transformer(adv_patch=perturbed_latent_vector,
                                                                  lab_batch=label,
                                                                  img_size=416,
                                                                  patch_mask=[],
                                                                  by_rectangle=by_rectangle,
                                                                  do_rotate=enable_rotation,
                                                                  rand_loc=enable_randomLocation,
                                                                  scale_rate=patch_scale,
                                                                  with_crease=enable_crease,
                                                                  with_projection=enable_projection,
                                                                  with_rectOccluding=enable_rectOccluding,
                                                                  enable_blurred=enable_blurred)
        p_img_batch = patch_applier(input_imgs, adv_batch_t)

        max_prob_obj_cls, output_class_target, bboxes = None, None, None
        if model_name == "yolov3" or model_name == "yolov2" or model_name == "yolov4":
            max_prob_obj_cls, output_class_target, bboxes = detector.detect(input_imgs=p_img_batch,
                                                                      cls_id_attacked=cls_id_attacked, epoch=epoch, with_bbox=enable_with_bbox)
        elif model_name == "yolov5":
            max_prob_obj_cls, bboxes = detector.detect(p_img_batch, label, cls_id_attacked)
            max_prob_obj_cls = torch.tensor([p for p in max_prob_obj_cls if p.numel() > 0], device=max_prob_obj_cls[0].device)
            max_prob_obj_cls = torch.tensor(0.0, device=latent_vector.device) if len(max_prob_obj_cls) == 0 else max_prob_obj_cls
            
        det_loss = torch.mean(max_prob_obj_cls)
        total_loss = det_loss
        
        if best_loss > total_loss:
            best_loss = total_loss
            best_latent = perturbed_latent_vector
            if enable_with_bbox and len(bboxes) > 0:
                best_p_img_batch = show_images_with_bbox(p_img_batch, bboxes, cls_id_attacked, model_name)

    return best_loss, torch.tensor(0), torch.tensor(0), torch.tensor(0), best_loss, best_p_img_batch


def train_square_attack(epoch, opt,
                        latent_vector, input_imgs,
                        label, patch_scale,
                        cls_id_attacked, model_name,
                        detector, patch_transformer,
                        patch_applier, total_variation,
                        by_rectangle, enable_rotation,
                        enable_randomLocation, enable_crease,
                        enable_projection, enable_rectOccluding,
                        enable_blurred, enable_with_bbox, weight_loss_tv, weight_loss_cls,
                        attack_name, classifier=None, classifier_weights=None):
    global best_loss, best_latent, best_p_img_batch
    with torch.no_grad():
        
        perturbed_latent_vector = update_square_attack_patch_l_inf(deepcopy(latent_vector))
        
        adv_batch_t, adv_patch_set, msk_batch = patch_transformer(adv_patch=perturbed_latent_vector,
                                                                  lab_batch=label,
                                                                  img_size=416,
                                                                  patch_mask=[],
                                                                  by_rectangle=by_rectangle,
                                                                  do_rotate=enable_rotation,
                                                                  rand_loc=enable_randomLocation,
                                                                  scale_rate=patch_scale,
                                                                  with_crease=enable_crease,
                                                                  with_projection=enable_projection,
                                                                  with_rectOccluding=enable_rectOccluding,
                                                                  enable_blurred=enable_blurred)
        p_img_batch = patch_applier(input_imgs, adv_batch_t)

        max_prob_obj_cls, output_class_target, bboxes = None, None, None
        if model_name == "yolov3" or model_name == "yolov2" or model_name == "yolov4":
            max_prob_obj_cls, output_class_target, bboxes = detector.detect(input_imgs=p_img_batch,
                                                                      cls_id_attacked=cls_id_attacked, epoch=epoch, with_bbox=enable_with_bbox)
        elif model_name == "yolov5":
            max_prob_obj_cls, bboxes = detector.detect(p_img_batch, label, cls_id_attacked)
            max_prob_obj_cls = torch.tensor([p for p in max_prob_obj_cls if p.numel() > 0], device=max_prob_obj_cls[0].device)
            max_prob_obj_cls = torch.tensor(0.0, device=latent_vector.device) if len(max_prob_obj_cls) == 0 else max_prob_obj_cls
            
        det_loss = torch.mean(max_prob_obj_cls)
        total_loss = det_loss
        
        if best_loss > total_loss:
            best_loss = total_loss
            best_latent = perturbed_latent_vector
            if enable_with_bbox and len(bboxes) > 0:
                best_p_img_batch = show_images_with_bbox(p_img_batch, bboxes, cls_id_attacked, model_name)

    return best_loss, torch.tensor(0), torch.tensor(0), torch.tensor(0), best_loss, best_p_img_batch


def train_random_patch_gan(epoch, generator
                    , opt, device
                    , latent_vector
                    , input_imgs, label, patch_scale, cls_id_attacked
                    , model_name, detector
                    , patch_transformer, patch_applier
                    , total_variation
                    , by_rectangle
                    , enable_rotation
                    , enable_randomLocation
                    , enable_crease
                    , enable_projection
                    , enable_rectOccluding
                    , enable_blurred
                    , enable_with_bbox
                    , weight_loss_tv
                    , weight_loss_cls
                        , attack_name
                        , classifier
                        , classifier_weights
                        , n_pop
                        , sigma):
    global best_loss, best_latent, best_p_img_batch
    with torch.no_grad():
        perturbed_latent_vector = latent_vector + torch.randn(latent_vector.shape).cuda() * 0.001
        interpolation_deformed = evo_interpolate_shift(generator, perturbed_latent_vector,
                                                   latent_vector=torch.zeros_like(perturbed_latent_vector.unsqueeze(0)).to(device))
        interpolation_deformed = interpolation_deformed.to(device)  # torch.Size([1, 3, 128, 128])
        fake_images = (interpolation_deformed + 1) * 0.5
        
        for i, patch in enumerate(fake_images):
            
            adv_batch_t, adv_patch_set, msk_batch = patch_transformer(adv_patch=patch,
                                                                    lab_batch=label,
                                                                    img_size=416,
                                                                    patch_mask=[],
                                                                    by_rectangle=by_rectangle,
                                                                    do_rotate=enable_rotation,
                                                                    rand_loc=enable_randomLocation,
                                                                    scale_rate=patch_scale,
                                                                    with_crease=enable_crease,
                                                                    with_projection=enable_projection,
                                                                    with_rectOccluding=enable_rectOccluding,
                                                                    enable_blurred=enable_blurred)
            p_img_batch = patch_applier(input_imgs, adv_batch_t)

            max_prob_obj_cls, output_class_target, bboxes = None, None, None
            if model_name == "yolov3" or model_name == "yolov2" or model_name == "yolov4":
                max_prob_obj_cls, output_class_target, bboxes = detector.detect(input_imgs=p_img_batch,
                                                                        cls_id_attacked=cls_id_attacked, epoch=epoch, with_bbox=enable_with_bbox)
            elif model_name == "yolov5":
                max_prob_obj_cls, bboxes = detector.detect(p_img_batch, label, cls_id_attacked)
                max_prob_obj_cls = torch.tensor([p for p in max_prob_obj_cls if p.numel() > 0], device=max_prob_obj_cls[0].device)
                max_prob_obj_cls = torch.tensor(0.0, device=latent_vector.device) if len(max_prob_obj_cls) == 0 else max_prob_obj_cls
                
            det_loss = torch.mean(max_prob_obj_cls)
            total_loss = det_loss
            tv_loss = total_variation(fake_images[0])
            cls_loss = classifier_predict(classifier, classifier_weights, patch.unsqueeze(0), generator.target_classes)
            cls_loss = (1 - cls_loss[0])
            total_loss = (weight_loss_tv * tv_loss) + det_loss + (weight_loss_cls * cls_loss)
            
            if best_loss > total_loss:
                best_loss = total_loss
                best_latent = perturbed_latent_vector
                if enable_with_bbox and len(bboxes) > 0:
                    best_p_img_batch = show_images_with_bbox(p_img_batch, bboxes, cls_id_attacked, model_name)

    return best_loss, tv_loss, cls_loss, torch.tensor(0), best_loss, best_p_img_batch


def train_gan_patch(epoch, generator
                    , opt, device
                    , latent_vector
                    , input_imgs, label, patch_scale, cls_id_attacked
                    , model_name, detector
                    , patch_transformer, patch_applier
                    , total_variation
                    , by_rectangle
                    , enable_rotation
                    , enable_randomLocation
                    , enable_crease
                    , enable_projection
                    , enable_rectOccluding
                    , enable_blurred
                    , enable_with_bbox
                    , weight_loss_tv
                    , weight_loss_cls
                    , attack_name=0
                    , classifier=None
                    , classifier_weights=None
                    , target_xai=None
                    ):
    opt.zero_grad()
    z = latent_vector
    interpolation_deformed = interpolate_shift(generator, z.unsqueeze(0),
                                               latent_vector=torch.zeros_like(z.unsqueeze(0)).to(device),
                                               device=device)
    interpolation_deformed = interpolation_deformed.unsqueeze(0).to(device)
    fake_images = (interpolation_deformed + 1) * 0.5

    adv_batch_t, adv_patch_set, msk_batch = patch_transformer(adv_patch=fake_images[0],
                                                              lab_batch=label,
                                                              img_size=416,
                                                              patch_mask=[],
                                                              by_rectangle=by_rectangle,
                                                              do_rotate=enable_rotation,
                                                              rand_loc=enable_randomLocation,
                                                              scale_rate=patch_scale,
                                                              with_crease=enable_crease,
                                                              with_projection=enable_projection,
                                                              with_rectOccluding=enable_rectOccluding,
                                                              enable_blurred=enable_blurred)
    p_img_batch = patch_applier(input_imgs, adv_batch_t)  # torch.Size([8, 14, 3, 416, 416])

    max_prob_obj_cls, output_class_target, bboxes = None, None, None
    if model_name == "yolov3" or model_name == "yolov2" or model_name == "yolov4":
        max_prob_obj_cls, output_class_target, bboxes = detector.detect(input_imgs=p_img_batch,
                                                                    cls_id_attacked=cls_id_attacked, epoch=epoch, with_bbox=enable_with_bbox)
    elif model_name == "yolov5":
        max_prob_obj_cls, bboxes = detector.detect(p_img_batch, label, cls_id_attacked)
        max_prob_obj_cls = torch.tensor([p for p in max_prob_obj_cls if p.numel() > 0], device=max_prob_obj_cls[0].device)
        max_prob_obj_cls = torch.tensor(0.0, device=latent_vector.device) if len(max_prob_obj_cls) == 0 else max_prob_obj_cls
        
    det_loss = torch.mean(max_prob_obj_cls)

    if attack_name == 'fn':
        tv_loss = total_variation(fake_images[0])
        total_loss = det_loss + (weight_loss_tv * tv_loss)
    elif attack_name == 'fp':
        tv_loss = total_variation(fake_images[0])
        cls_loss = classifier_predict(classifier, classifier_weights, fake_images, generator.target_classes)
        cls_loss = (1 - cls_loss)
        det_loss = (1 - det_loss)
        total_loss = (weight_loss_tv * tv_loss) + det_loss + (weight_loss_cls * cls_loss)
    elif attack_name == 'xai':
        tv_loss = total_variation(fake_images[0])
        cls_loss = classifier_predict(classifier, classifier_weights, fake_images, generator.target_classes)
        cls_loss = (1 - cls_loss)
        target_layers = [detector.model.module_list[-2][0]]
        cam = EigenCAM(detector.model, target_layers, use_cuda=False)
        xais = torch.tensor(cam(p_img_batch)).to(device)
        save_image(xais[0], 'curr_xai.png')
        xai_loss = F.mse_loss(xais, target_xai.repeat(p_img_batch.shape[0], 1, 1))
        total_loss = (weight_loss_tv * tv_loss) + det_loss + (weight_loss_cls * cls_loss) + xai_loss

    total_loss.backward()
    opt.step()

    # draw bbox at output
    if enable_with_bbox and len(bboxes) > 0:
        p_img_batch = show_images_with_bbox(p_img_batch, bboxes, cls_id_attacked, model_name)
    if attack_name == 'fp':
        return det_loss, tv_loss, cls_loss, torch.tensor(0), total_loss, p_img_batch
    else:
        return det_loss, tv_loss, torch.tensor(0), xai_loss, total_loss, p_img_batch
    
def evo_train_gan_patch(epoch, generator
                    , opt, device
                    , latent_vector
                    , input_imgs, label, patch_scale, cls_id_attacked
                    , model_name, detector
                    , patch_transformer, patch_applier
                    , total_variation
                    , by_rectangle
                    , enable_rotation
                    , enable_randomLocation
                    , enable_crease
                    , enable_projection
                    , enable_rectOccluding
                    , enable_blurred
                    , enable_with_bbox
                    , weight_loss_tv
                    , weight_loss_cls
                        , attack_name
                        , classifier
                        , classifier_weights
                        , n_pop
                        , sigma):
    with torch.no_grad():
        opt.zero_grad()
        z = latent_vector

        dist = torch.randn((n_pop, latent_vector.shape[0])).cuda()

        pop = z + (dist * sigma)

        interpolation_deformed = evo_interpolate_shift(generator, pop,
                                                   latent_vector=torch.zeros_like(z.unsqueeze(0)).to(device))
        interpolation_deformed = interpolation_deformed.to(device)  # torch.Size([1, 3, 128, 128])
        fake_images = (interpolation_deformed + 1) * 0.5

        det_losses = torch.zeros(n_pop)
        tv_losses = torch.zeros(n_pop)
        cls_losses = torch.zeros(n_pop)
        total_losses = torch.zeros(n_pop)


        for i, patch in enumerate(fake_images):
            adv_batch_t, adv_patch_set, msk_batch = patch_transformer(adv_patch=patch,
                                                                      lab_batch=label,
                                                                      img_size=416,
                                                                      patch_mask=[],
                                                                      by_rectangle=by_rectangle,
                                                                      do_rotate=enable_rotation,
                                                                      rand_loc=enable_randomLocation,
                                                                      scale_rate=patch_scale,
                                                                      with_crease=enable_crease,
                                                                      with_projection=enable_projection,
                                                                      with_rectOccluding=enable_rectOccluding,
                                                                      enable_blurred=enable_blurred)
            p_img_batch = patch_applier(input_imgs, adv_batch_t)  # torch.Size([8, 14, 3, 416, 416])
            
            max_prob_obj_cls, output_class_target, bboxes = None, None, None
            if model_name == "yolov3" or model_name == "yolov2" or model_name == "yolov4":
                max_prob_obj_cls, output_class_target, bboxes = detector.detect(input_imgs=p_img_batch,
                                                                            cls_id_attacked=cls_id_attacked, epoch=epoch, with_bbox=enable_with_bbox)
            elif model_name == "yolov5":
                max_prob_obj_cls, bboxes = detector.detect(p_img_batch, label, cls_id_attacked)
                max_prob_obj_cls = torch.tensor([p for p in max_prob_obj_cls if p.numel() > 0], device=max_prob_obj_cls[0].device)
                max_prob_obj_cls = torch.tensor(0.0, device=latent_vector.device) if len(max_prob_obj_cls) == 0 else max_prob_obj_cls
                
            loss_det = torch.mean(max_prob_obj_cls)

            if attack_name == 'fn':
                tv_losses[i] = total_variation(fake_images[i])
                loss_cls = classifier_predict(classifier, classifier_weights, patch.unsqueeze(0), generator.target_classes)
                loss_cls = (1 - loss_cls)
                det_losses[i] = loss_det
                cls_losses[i] = loss_cls
                total_losses[i] = (weight_loss_tv * tv_losses[i]) + det_losses[i] + (weight_loss_cls * cls_losses[i])
            elif attack_name == 'fp':
                tv_losses[i] = total_variation(fake_images[i])
                loss_cls = classifier_predict(classifier, classifier_weights, patch.unsqueeze(0), generator.target_classes)
                loss_cls = (1 - loss_cls)
                loss_det = (1 - loss_det)
                det_losses[i] = loss_det
                cls_losses[i] = loss_cls
                total_losses[i] = (weight_loss_tv * tv_losses[i]) + det_losses[i] + (weight_loss_cls * cls_losses[i])

        loss = (total_losses - total_losses.mean()) / total_losses.std()
        loss = loss.to(device)
        pop = pop.to(device)

        latent_vector.grad = pop.T @ loss / (n_pop * sigma)
        opt.step()

        if enable_with_bbox and len(bboxes) > 0:
            p_img_batch = show_images_with_bbox(p_img_batch, bboxes, cls_id_attacked, model_name)

    return det_losses.mean(), tv_losses.mean(), cls_losses.mean(), torch.tensor(0), total_losses.mean(), p_img_batch

def nes_est_train_patch(epoch, generator
                    , opt, device
                    , latent_vector
                    , input_imgs, label, patch_scale, cls_id_attacked
                    , model_name, detector
                    , patch_transformer, patch_applier
                    , total_variation
                    , by_rectangle
                    , enable_rotation
                    , enable_randomLocation
                    , enable_crease
                    , enable_projection
                    , enable_rectOccluding
                    , enable_blurred
                    , enable_with_bbox
                    , weight_loss_tv
                    , weight_loss_cls
                        , attack_name
                        , classifier
                        , classifier_weights
                        , n_pop
                        , sigma):
    with torch.no_grad():
        opt.zero_grad()
        z = latent_vector

        dist = torch.randn((n_pop, latent_vector.shape[0], latent_vector.shape[1], latent_vector.shape[2])).cuda()

        pop1 = z + (dist * sigma)
        pop2 = z - (dist * sigma)
        pop = torch.vstack([pop1, pop2])

        # interpolation_deformed = evo_interpolate_shift(generator, pop,
        #                                            latent_vector=torch.zeros_like(z.unsqueeze(0)).to(device))
        # interpolation_deformed = interpolation_deformed.to(device)  # torch.Size([1, 3, 128, 128])
        # fake_images = (interpolation_deformed + 1) * 0.5

        det_losses = torch.zeros(n_pop * 2)
        tv_losses = torch.zeros(n_pop * 2)
        cls_losses = torch.zeros(n_pop * 2)
        total_losses = torch.zeros(n_pop * 2)


        for i, patch in enumerate(pop):
            adv_batch_t, adv_patch_set, msk_batch = patch_transformer(adv_patch=patch,
                                                                      lab_batch=label,
                                                                      img_size=416,
                                                                      patch_mask=[],
                                                                      by_rectangle=by_rectangle,
                                                                      do_rotate=enable_rotation,
                                                                      rand_loc=enable_randomLocation,
                                                                      scale_rate=patch_scale,
                                                                      with_crease=enable_crease,
                                                                      with_projection=enable_projection,
                                                                      with_rectOccluding=enable_rectOccluding,
                                                                      enable_blurred=enable_blurred)
            p_img_batch = patch_applier(input_imgs, adv_batch_t)  # torch.Size([8, 14, 3, 416, 416])

            max_prob_obj_cls, output_class_target, bboxes = None, None, None
            if model_name == "yolov3" or model_name == "yolov2" or model_name == "yolov4":
                max_prob_obj_cls, output_class_target, bboxes = detector.detect(input_imgs=p_img_batch,
                                                                            cls_id_attacked=cls_id_attacked, epoch=epoch, with_bbox=enable_with_bbox)
            elif model_name == "yolov5":
                max_prob_obj_cls, bboxes = detector.detect(p_img_batch, label, cls_id_attacked)
                max_prob_obj_cls = torch.tensor([p for p in max_prob_obj_cls if p.numel() > 0], device=max_prob_obj_cls[0].device)
                max_prob_obj_cls = torch.tensor(0.0, device=latent_vector.device) if len(max_prob_obj_cls) == 0 else max_prob_obj_cls
        
            loss_det = torch.mean(max_prob_obj_cls)

            if attack_name == 'fn':
                # tv_losses[i] = total_variation(pop[i])
                # loss_cls = classifier_predict(classifier, classifier_weights, patch.unsqueeze(0), generator.target_classes)
                # loss_cls = (1 - loss_cls)
                det_losses[i] = loss_det
                # cls_losses[i] = loss_cls
                # total_losses[i] = (weight_loss_tv * tv_losses[i]) + det_losses[i] + (weight_loss_cls * cls_losses[i])
                total_losses[i] = det_losses[i]
                
        loss = (total_losses - total_losses.mean()) / total_losses.std()
        loss = loss.to(device)
        pop = pop.to(device)
        
        grad = torch.zeros_like(latent_vector)
        grad += (pop[:n_pop].T @ loss[:n_pop]).T
        grad -= (pop[n_pop:].T @ loss[n_pop:]).T

        latent_vector.grad = grad / (2 * n_pop * sigma)
        opt.step()

        if enable_with_bbox and len(bboxes) > 0:
            p_img_batch = show_images_with_bbox(p_img_batch, bboxes, cls_id_attacked, model_name)

    return det_losses.mean(), tv_losses.mean(), cls_losses.mean(), torch.tensor(0), total_losses.mean(), p_img_batch

def nes_train_patch(epoch, generator
                    , opt, device
                    , latent_vector
                    , input_imgs, label, patch_scale, cls_id_attacked
                    , model_name, detector
                    , patch_transformer, patch_applier
                    , total_variation
                    , by_rectangle
                    , enable_rotation
                    , enable_randomLocation
                    , enable_crease
                    , enable_projection
                    , enable_rectOccluding
                    , enable_blurred
                    , enable_with_bbox
                    , weight_loss_tv
                    , weight_loss_cls
                        , attack_name
                        , classifier
                        , classifier_weights
                        , n_pop
                        , sigma
                        , target_xai):
    with torch.no_grad():
        opt.zero_grad()
        z = latent_vector

        dist = torch.randn((n_pop, latent_vector.shape[0])).cuda()

        pop = z + (dist * sigma)

        interpolation_deformed = evo_interpolate_shift(generator, pop,
                                                   latent_vector=torch.zeros_like(z.unsqueeze(0)).to(device))
        interpolation_deformed = interpolation_deformed.to(device)  # torch.Size([1, 3, 128, 128])
        fake_images = (interpolation_deformed + 1) * 0.5

        xai_loss = torch.tensor(0)

        pop_p_img_batch = []
        for i, patch in enumerate(fake_images):
            adv_batch_t, adv_patch_set, msk_batch = patch_transformer(adv_patch=patch,
                                                                      lab_batch=label,
                                                                      img_size=416,
                                                                      patch_mask=[],
                                                                      by_rectangle=by_rectangle,
                                                                      do_rotate=enable_rotation,
                                                                      rand_loc=enable_randomLocation,
                                                                      scale_rate=patch_scale,
                                                                      with_crease=enable_crease,
                                                                      with_projection=enable_projection,
                                                                      with_rectOccluding=enable_rectOccluding,
                                                                      enable_blurred=enable_blurred)
            p_img_batch = patch_applier(input_imgs, adv_batch_t)  # torch.Size([8, 14, 3, 416, 416])
            pop_p_img_batch.append(p_img_batch)

        pop_p_img_batch = torch.vstack(pop_p_img_batch)
        if '3090' in torch.cuda.get_device_name(0) and n_pop > 60:
            truncated_batch_size = pop_p_img_batch.shape[0] // 2
            batch_max_prob_obj_cls, batch_output_class_target, batch_bboxes = [], [], []
            for i in range(2):
                
                max_prob_obj_cls, output_class_target, bboxes = None, None, None
                if model_name == "yolov3" or model_name == "yolov2" or model_name == "yolov4":
                    max_prob_obj_cls, output_class_target, bboxes = detector.detect(input_imgs=p_img_batch,
                                                                                cls_id_attacked=cls_id_attacked, epoch=epoch, with_bbox=enable_with_bbox)
                elif model_name == "yolov5":
                    max_prob_obj_cls, bboxes = detector.detect(p_img_batch, label, cls_id_attacked)
                    max_prob_obj_cls = torch.tensor([p for p in max_prob_obj_cls if p.numel() > 0], device=max_prob_obj_cls[0].device)
                    max_prob_obj_cls = torch.tensor(0.0, device=latent_vector.device) if len(max_prob_obj_cls) == 0 else max_prob_obj_cls
        
                batch_max_prob_obj_cls.append(max_prob_obj_cls)
                batch_bboxes.append(bboxes)
            max_prob_obj_cls = torch.cat(batch_max_prob_obj_cls, dim=0)
            bboxes = [item for sublist in batch_bboxes for item in sublist]
        else:
            max_prob_obj_cls, output_class_target, bboxes = detector.detect(input_imgs=pop_p_img_batch,cls_id_attacked=cls_id_attacked, epoch=epoch, with_bbox=enable_with_bbox)

        det_chunks = max_prob_obj_cls.chunk(n_pop)
        det_loss = torch.tensor([chunk.mean() for chunk in det_chunks]).cuda()

        if attack_name == 'fn':
            tv_loss = total_variation(fake_images)
            cls_loss = classifier_predict(classifier, classifier_weights, fake_images, generator.target_classes)
            cls_loss = (1 - cls_loss)
            total_loss = (weight_loss_tv * tv_loss) + det_loss + (weight_loss_cls * cls_loss)

        elif attack_name == 'fp':
            tv_loss = total_variation(fake_images)
            cls_loss = classifier_predict(classifier, classifier_weights, fake_images, generator.target_classes)
            cls_loss = (1 - cls_loss)
            det_loss = (1 - det_loss)
            total_loss = (weight_loss_tv * tv_loss) + det_loss + (weight_loss_cls * cls_loss)

        elif attack_name == 'xai':
            tv_loss = total_variation(fake_images)
            cls_loss = classifier_predict(classifier, classifier_weights, fake_images, generator.target_classes)
            cls_loss = (1 - cls_loss)
            target_layers = [detector.model.module_list[-2][0]]
            cam = EigenCAM(detector.model, target_layers, use_cuda=False)
            indices = torch.randint(low=0, high=pop_p_img_batch.shape[0], size=(50,))
            xais = torch.tensor(cam(pop_p_img_batch[indices, :, :, :])).to(device)
            save_image(target_xai, 'target_xai.png')
            xai_loss = F.mse_loss(xais, target_xai.repeat(50, 1, 1))
            print(f'Xai loss: {xai_loss:.4f}')
            save_image(xais[0], 'curr_xai.png')
            # total_loss = (weight_loss_tv * tv_loss) + det_loss + (weight_loss_cls * cls_loss) + xai_loss
            total_loss = (weight_loss_tv * tv_loss) + (weight_loss_cls * cls_loss) + xai_loss

        else:
            raise Exception(f'No such attack {attack_name}')

        loss = (total_loss - total_loss.mean()) / total_loss.std()
        loss = loss.to(device)
        pop = pop.to(device)

        latent_vector.grad = pop.T @ loss / (n_pop * sigma)
        opt.step()

        if enable_with_bbox and len(bboxes) > 0:
            p_img_batch = show_images_with_bbox(p_img_batch, bboxes, cls_id_attacked, model_name)

    return det_loss.mean(), tv_loss.mean(), cls_loss.mean(), xai_loss, total_loss.mean(), p_img_batch