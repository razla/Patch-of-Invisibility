import torch
import torch.nn as nn

from GANLatentDiscovery.visualization import evo_interpolate_shift, interpolate_shift
from new_utils import classifier_predict, show_images_with_bbox


class TotalVariation(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total vatiation (TV) of an patch.

    """

    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, adv_patch):
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1] + 0.000001), 0)
        tvcomp1 = torch.sum(torch.sum(tvcomp1, 0), 0)
        tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :] + 0.000001), 0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2, 0), 0)
        tv = tvcomp1 + tvcomp2
        return tv / torch.numel(adv_patch)

def train_rowPtach(method_num, epoch, generator
                   , opt, device
                   , latent_shift
                   , input_imgs, label, true_cls, patch_scale, cls_id_attacked
                   , model_name
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
                   , weight_loss_pred
                   , max_value_latent_item=8
                   , attack_num=0
                   , classifier=None
                   , classifier_weights=None
                   ):
    # raw patch with grad
    if method_num == 0:
        return train_raw_patch(epoch, opt,
                               latent_shift, input_imgs,
                               label, true_cls, patch_scale,
                               cls_id_attacked, model_name,
                            patch_transformer,
                               patch_applier, total_variation,
                               by_rectangle, enable_rotation,
                               enable_randomLocation, enable_crease,
                               enable_projection, enable_rectOccluding,
                               enable_blurred, enable_with_bbox, weight_loss_tv, weight_loss_pred, attack_num,
                               classifier, classifier_weights)
    # gan patch without grad
    if method_num == 1:
        return evo_train_gan_patch(epoch, generator
                                   , opt, device
                                   , latent_shift
                                   , input_imgs, label, true_cls, patch_scale, cls_id_attacked
                                   , model_name
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
                                   , weight_loss_pred
                                   , max_value_latent_item
                                   , attack_num
                                   , classifier
                                   , classifier_weights)
    # gan patch with grad
    if method_num == 2:  # biggan
        return train_gan_patch(epoch, generator
                               , opt, device
                               , latent_shift
                               , input_imgs, label, true_cls, patch_scale, cls_id_attacked
                               , model_name
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
                               , weight_loss_pred
                               , max_value_latent_item=8)


def train_raw_patch(epoch, opt,
                    latent_shift, input_imgs,
                    label, true_cls, patch_scale,
                    cls_id_attacked, model_name,
                    patch_transformer,
                    patch_applier, total_variation,
                    by_rectangle, enable_rotation,
                    enable_randomLocation, enable_crease,
                    enable_projection, enable_rectOccluding,
                    enable_blurred, enable_with_bbox, weight_loss_tv, weight_loss_pred, attack_num, classifier, classifier_weights):
    opt.zero_grad()
    adv_batch_t, adv_patch_set, msk_batch = patch_transformer(adv_patch=latent_shift,
                                                              lab_batch=label,
                                                              img_size=224,
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

    max_prob = classifier_predict(classifier, classifier_weights, p_img_batch, true_cls)
    loss_det = torch.mean(max_prob)

    loss_tv = total_variation(latent_shift)
    if attack_num == 0:
        loss = loss_det + (weight_loss_tv * loss_tv)
    elif attack_num == 1:
        loss_det = (1 - loss_det)
        loss = loss_det + (weight_loss_tv * loss_tv)

    loss.backward()
    opt.step()

    return loss_det, loss_tv, p_img_batch


def train_gan_patch(epoch, generator
                    , opt, device
                    , latent_shift
                    , input_imgs, label, true_cls, patch_scale, cls_id_attacked
                    , model_name
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
                    , weight_loss_pred
                    , max_value_latent_item=8, attack_num=0):
    opt.zero_grad()
    latent_shift.data = torch.clamp(latent_shift.data, -max_value_latent_item, max_value_latent_item)
    z = latent_shift
    interpolation_deformed = interpolate_shift(generator, z.unsqueeze(0),
                                               latent_shift=torch.zeros_like(z.unsqueeze(0)).to(device),
                                               device=device)
    interpolation_deformed = interpolation_deformed.unsqueeze(0).to(device)  # torch.Size([1, 3, 128, 128])
    # print("dore",interpolation_deformed[0,0,:10,0])
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

    # loss.
    if (model_name == "yolov4" or model_name == "yolov3" or model_name == "yolov2"):
        max_prob_obj_cls, output_class_target, overlap_score, bboxes = detector.detect(input_imgs=p_img_batch,
                                                                                       cls_id_attacked=cls_id_attacked,
                                                                                       epoch=epoch,
                                                                                       clear_imgs=input_imgs,
                                                                                       with_bbox=enable_with_bbox)
        loss_det = torch.mean(max_prob_obj_cls)

    loss_tv = total_variation(fake_images[0])
    if attack_num == 0:
        loss = loss_det + (weight_loss_tv * loss_tv)
    elif attack_num == 1:
        loss_det = (1 - loss_det)
        loss = loss_det + (weight_loss_tv * loss_tv)
    elif attack_num == 2:
        loss_det = (1 - torch.mean(output_class_target))
        loss = loss_det + (weight_loss_tv * loss_tv)

    loss.backward()
    opt.step()

    # draw bbox at output
    if enable_with_bbox and len(bboxes) > 0 and (epoch % detector.epoch_save == 0):
        p_img_batch = show_images_with_bbox(p_img_batch, bboxes, cls_id_attacked, model_name)
    return loss_det, loss_tv, p_img_batch


def evo_train_gan_patch(epoch, generator
                        , opt, device
                        , latent_shift
                        , input_imgs, label, true_cls, patch_scale, cls_id_attacked
                        , model_name
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
                        , weight_loss_pred
                        , max_value_latent_item=8
                        , attack_num=0
                        , classifier=None
                        , classifier_weights=None
                        , n_pop=30
                        , sigma=0.1):
    with torch.no_grad():
        opt.zero_grad()
        latent_shift.data = torch.clamp(latent_shift.data, -max_value_latent_item, max_value_latent_item)
        z = latent_shift

        dist = torch.randn((n_pop, latent_shift.shape[0])).cuda()

        pop = z + (dist * sigma)

        interpolation_deformed = evo_interpolate_shift(generator, pop,
                                                       latent_shift=torch.zeros_like(z.unsqueeze(0)).to(device))
        interpolation_deformed = interpolation_deformed.to(device)  # torch.Size([1, 3, 128, 128])
        fake_images = (interpolation_deformed + 1) * 0.5

        det_losses = torch.zeros(n_pop)
        tv_losses = torch.zeros(n_pop)
        overall_losses = torch.zeros(n_pop)

        for i, patch in enumerate(fake_images):
            adv_batch_t, adv_patch_set, msk_batch = patch_transformer(adv_patch=patch,
                                                                      lab_batch=label,
                                                                      img_size=224,
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

            # loss.
            if (model_name == "yolov4" or model_name == "yolov3" or model_name == "yolov2"):
                max_prob_obj_cls, output_class_target, overlap_score, bboxes = detector.detect(input_imgs=p_img_batch,
                                                                                               cls_id_attacked=cls_id_attacked,
                                                                                               clear_imgs=input_imgs,
                                                                                               epoch=epoch,
                                                                                               with_bbox=enable_with_bbox)
                loss_det = torch.mean(max_prob_obj_cls)

            tv_losses[i] = total_variation(fake_images[i])
            if attack_num == 0:
                det_losses[i] = loss_det
                overall_losses[i] = (weight_loss_tv * tv_losses[i]) + det_losses[i]
            elif attack_num == 1:
                loss_pred = classifier_predict(classifier, classifier_weights, patch, generator.target_classes)
                loss_pred = (1 - loss_pred) * weight_loss_pred
                loss_det = (1 - loss_det)
                det_losses[i] = loss_det + loss_pred
                overall_losses[i] = (weight_loss_tv * tv_losses[i]) + det_losses[i]
            elif attack_num == 2:
                det_losses[i] = (1 - torch.mean(output_class_target))
                overall_losses[i] = (weight_loss_tv * tv_losses[i]) + det_losses[i]

        loss = (overall_losses - overall_losses.mean()) / overall_losses.std()
        loss = loss.to(device)
        pop = pop.to(device)

        latent_shift.grad = pop.T @ loss / (n_pop * sigma)
        opt.step()

        if enable_with_bbox and len(bboxes) > 0 and (epoch % detector.epoch_save == 0):
            p_img_batch = show_images_with_bbox(p_img_batch, bboxes, cls_id_attacked, model_name)
    return det_losses.mean(), tv_losses.mean(), p_img_batch