from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from PIL import ImageDraw, ImageFont
# from lion_pytorch import Lion
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
import shutil
import torch
import time
import os

from pytorchYOLOv4.tool.utils import load_class_names
from adversarialYolo.load_data import InriaDataset
from adversarialYolo.load_data import ImageNetDataset
from GANLatentDiscovery.loading import load_from_dir


def load_generator(method_name):
    generator = None
    len_z = None
    len_latent = None
    if (method_name == 'bbgan' or method_name == 'wbgan' or method_name == 'nes' or method_name == 'random_gan'):
        _, G, _ = load_from_dir(
            'GANLatentDiscovery/models/pretrained/deformators/BigGAN/',
            G_weights='./GANLatentDiscovery/models/pretrained/generators/BigGAN/G_ema.pth')
        generator = G
        len_z = G.dim_z
        len_latent = G.dim_z
        print("setting: len_latent : " + str(len_latent))
    return generator, len_z, len_latent

def load_detector(model_name, tiny, epoch_save, multi_score):
    start = time.time()
    if (model_name == "yolov2"):
        from adversarialYolo.demo import DetectorYolov2
        detector = DetectorYolov2(show_detail=False, tiny=tiny, epoch_save=epoch_save, multi_score=multi_score)
    elif (model_name == "yolov3"):
        from PyTorchYOLOv3.detect import DetectorYolov3
        detector = DetectorYolov3(show_detail=False, tiny=tiny, epoch_save=epoch_save, multi_score=multi_score)
    elif (model_name == "yolov4"):
        from pytorchYOLOv4.demo import DetectorYolov4
        detector = DetectorYolov4(show_detail=False, tiny=tiny, epoch_save=epoch_save, multi_score=multi_score)
    elif (model_name == "yolov5"):
        from PyTorchYOLOv5.detect import DetectorYolov5
        detector = DetectorYolov5(tiny=tiny, multi_score=multi_score)
    elif (model_name == 'ssd3'):
        from Detectors.ssd3 import SSD3
        detector = SSD3(show_detail=False, tiny=tiny, epoch_save=epoch_save, multi_score=multi_score)
    elif (model_name == 'ssd'):
        from Detectors.ssd import SSD
        detector = SSD(show_detail=False, tiny=tiny, epoch_save=epoch_save, multi_score=multi_score)
    elif (model_name == 'frcnn'):
        from Detectors.fasterrcnn import FRCNN
        detector = FRCNN(show_detail=False, tiny=tiny, epoch_save=epoch_save, multi_score=multi_score)
    else:
        raise Exception(f'No such detector {model_name}')
    finish = time.time()
    print('Load detector in %f seconds.' % (finish - start))
    return detector

def load_dataset(dataset, batch_size, attack_name, train=True):
    ds_image_size = 416 if not dataset == 'minivan' else 224
    if dataset == "inria" and attack_name != 'fp':
        train_loader = torch.utils.data.DataLoader(
            InriaDataset(img_dir='/cs_storage/public_datasets/inria/inria/Train/pos',
                         lab_dir='/cs_storage/public_datasets/inria/inria/Train/pos/yolo-labels_yolov3tiny',
                         max_lab=14 if train else 1,
                         imgsize=ds_image_size,
                         shuffle=True),
            batch_size=batch_size,
            shuffle=True,
            num_workers=6)
    if dataset == 'one_person' and attack_name != 'fp':
        train_loader = torch.utils.data.DataLoader(
            InriaDataset(img_dir='/cs_storage/public_datasets/inria/inria/Train/pos/one_person',
                         lab_dir='/cs_storage/public_datasets/inria/inria/Train/pos/one_person/one_person_labels',
                         max_lab=14 if train else 1,
                         imgsize=ds_image_size,
                         shuffle=True),
            batch_size=batch_size,
            shuffle=True,
            num_workers=6)
    elif attack_name == 'fp':
        train_loader = torch.utils.data.DataLoader(
            InriaDataset(img_dir='/home/razla/inria/neg',
                         lab_dir='/home/razla/inria/yolo-labels_yolov3tiny',
                         max_lab=14 if train else 1,
                         imgsize=ds_image_size,
                         shuffle=True),
            batch_size=batch_size,
            shuffle=True,
            num_workers=6)
    elif dataset == 'test':
        train_loader = torch.utils.data.DataLoader(
            InriaDataset(img_dir='/home/razla/NatAdvPatcj/adversarialYolo/test/img',
                         lab_dir='/home/razla/NatAdvPatcj/adversarialYolo/test/lab',
                         max_lab=14,
                         imgsize=ds_image_size,
                         shuffle=True),
            batch_size=batch_size,
            shuffle=True,
            num_workers=6)
    elif dataset == 'ron':
        train_loader = torch.utils.data.DataLoader(
            InriaDataset(img_dir='/home/razla/PhyAdvPatch/Movie',
                         lab_dir='/home/razla/PhyAdvPatch/Movie/labels',
                         max_lab=14,
                         imgsize=ds_image_size,
                         shuffle=True),
            batch_size=batch_size,
            shuffle=True,
            num_workers=6)
    elif dataset == 'elbit':
        train_loader = torch.utils.data.DataLoader(
            InriaDataset(img_dir="/home/razla/elbit/val/images",
                         lab_dir="/home/razla/elbit/val/labels",
                         max_lab=40,
                         imgsize=ds_image_size,
                         shuffle=True),
            batch_size=batch_size,
            shuffle=True,
            num_workers=6)
    elif dataset == 'minivan':
        train_loader = torch.utils.data.DataLoader(
            ImageNetDataset(img_dir="/cs_storage/public_datasets/ImageNet/train/n03769881",
                         lab_dir="/cs_storage/public_datasets/minivan",
                         max_lab=14,
                         imgsize=ds_image_size,
                         shuffle=True),
            batch_size=batch_size,
            shuffle=True,
            num_workers=6
        )
    return train_loader

def load_classifier(model_name, device):
    match model_name:
        case 'resnet50':
            weights = ResNet50_Weights.DEFAULT
            model = resnet50(weights=weights).eval().to(device)
        case _:
            raise Exception(f'No such classifier {model_name}')

    # img = read_image("/cs_storage/razla/NatAdvPatcj/exp/exp14/generated/generated-images-0001.png")
    # # Step 2: Initialize the inference transforms
    # preprocess = weights.transforms()
    #
    # # Step 3: Apply inference preprocessing transforms
    # batch = preprocess(img).unsqueeze(0).to(device)
    #
    # # Step 4: Use the model and print the predicted category
    # prediction = model(batch).squeeze(0).softmax(0)
    # class_id = prediction.argmax().item()
    # score = prediction[class_id].item()
    # category_name = weights.meta["categories"][class_id]
    # print(f"{category_name}: {100 * score:.1f}%")
    return model, weights

def load_optimizer(opt_name, patch, lr):
    match opt_name:
        case 'adam':
            opt = torch.optim.Adam([patch], lr=lr, betas=(0.5, 0.999), amsgrad=True)
        case 'lion':
            opt = Lion([patch], lr=lr)
        case _:
            raise Exception(f'No such optimizer {opt_name}')
    return opt

def classifier_predict(model, weights, img, label):
    preprocess = weights.transforms()
    batch = preprocess(img)
    if batch.shape[0] == 1:
        prediction = model(batch).softmax(1)
    else:
        prediction = model(batch).softmax(0)
    score = prediction[:, label].squeeze(1)
    return score

def create_labels_files(imgs_folder, output_folder):
    counter = 0
    for file in os.listdir(imgs_folder):
        counter += 1
        name = file.split('.')[0]
        f = open(f"{output_folder}/{name}.txt", "w")
        x = random.uniform(0.3, 0.6)
        y = random.uniform(0.3, 0.6)
        width = random.uniform(0.09, 0.11)
        height = random.uniform(0.3, 0.5)

        f.write(f"0 {x:.6f} {y:.6f} {width:.6f} {height:.6f}")
        f.close()
    print(counter)

def create_labels_files_classification(imgs_folder, output_folder):
    counter = 0
    for file in os.listdir(imgs_folder):
        counter += 1
        name = file.split('.')[0]
        f = open(f"{output_folder}/{name}.txt", "w")
        x = random.uniform(0.3, 0.6)
        y = random.uniform(0.3, 0.6)
        width = random.uniform(0.09, 0.11)
        height = random.uniform(0.3, 0.5)

        f.write(f"0 {x:.6f} {y:.6f} {width:.6f} {height:.6f}")
        f.close()
    print(counter)

def show_images_with_bbox(p_img_batch, bboxes, cls_id_attacked, model_name):
    trans_2pilimage = transforms.ToPILImage()
    batch = p_img_batch.size()[0]
    namesfile = 'pytorchYOLOv4/data/coco.names'
    class_names = load_class_names(namesfile)
    for b in range(batch):
        img_pil = trans_2pilimage(p_img_batch[b].cpu())
        img_width = img_pil.size[0]
        img_height = img_pil.size[1]
        bbox = bboxes[b]
        for box in bbox:
            cls_id = box[6].int().item()
            cls_name = class_names[cls_id]
            cls_conf = box[5].item()
            if (cls_id == cls_id_attacked):
                if (model_name == "yolov2"):
                    x_center = box[0]
                    y_center = box[1]
                    width = box[2]
                    height = box[3]
                    left = (x_center.item() - width.item() / 2) * img_width
                    right = (x_center.item() + width.item() / 2) * img_width
                    top = (y_center.item() - height.item() / 2) * img_height
                    bottom = (y_center.item() + height.item() / 2) * img_height
                if (model_name == "yolov4") or (model_name == "yolov3"):
                    left = int(box[0] * img_width)
                    right = int(box[2] * img_width)
                    top = int(box[1] * img_height)
                    bottom = int(box[3] * img_height)
                if model_name == "yolov5":
                    left = int(box[0].item())
                    right = int(box[2].item())
                    top = int(box[1].item())
                    bottom = int(box[3].item())
                if (model_name == 'ssd3'):
                    left = int(box[1])
                    right = int(box[3])
                    top = int(box[0])
                    bottom = int(box[2])
                # img with prediction
                draw = ImageDraw.Draw(img_pil)
                shape = [left, top, right, bottom]
                draw.rectangle(shape, outline="blue")
                # text
                color = [0, 0, 255]
                font = ImageFont.truetype("cmb10.ttf", int(min(img_width, img_height) / 30))
                sentence = str(cls_name) + " (" + str(round(float(cls_conf), 2)) + ")"
                position = [left, top]
                draw.text(tuple(position), sentence, tuple(color), font=font)
        trans_2tensor = transforms.ToTensor()
        p_img_batch[b] = trans_2tensor(img_pil)
    return p_img_batch

def create_plots(det_losses, tv_losses, cls_losses, total_losses, path):
    fig, axs = plt.subplots(2, 2)
    x = np.arange(len(det_losses))
    axs[0, 0].plot(x, det_losses)
    axs[0, 0].set_title("Detection loss")
    axs[1, 0].plot(x, tv_losses)
    axs[1, 0].set_title("TV loss")
    axs[1, 0].sharex(axs[0, 0])
    axs[0, 1].plot(x, cls_losses)
    axs[0, 1].set_title("Classification loss")
    axs[1, 1].plot(x, total_losses)
    axs[1, 1].set_title("Total loss")
    axs[1, 1].sharex(axs[0, 1])
    fig.tight_layout()
    plt.savefig(f'{path}/graphs.png')

def filter_dataset_one_person(imgs_folder, labels_folder, imgs_target_folder, labels_target_folder):
    counter = 0
    imgs_files = []
    labels_files = []
    for file_path in os.listdir(labels_folder):
        file = open(os.path.join(labels_folder, file_path), 'r')
        if (len(file.readlines()) == 1):
            labels_files.append(file_path)
            img_path = file_path.split('.')[0] + '.png'
            imgs_files.append(img_path)
            counter += 1
    print(counter)
    for img_name, label_name in zip(imgs_files, labels_files):
        full_path_img = os.path.join(imgs_folder, img_name)
        full_path_label = os.path.join(labels_folder, label_name)
        shutil.copy(full_path_img, imgs_target_folder)
        shutil.copy(full_path_label, labels_target_folder)
    return imgs_files, labels_files

if __name__ == '__main__':
    imgs_path = "/cs_storage/public_datasets/inria/inria/Train/pos"
    labels_path = "/cs_storage/public_datasets/inria/inria/Train/pos/yolo-labels_yolov3tiny"
    target_imgs_path = "/cs_storage/public_datasets/inria/inria/Train/pos/one_person"
    target_labels_path = "/cs_storage/public_datasets/inria/inria/Train/pos/one_person/one_person_labels"
    filter_dataset_one_person(imgs_path, labels_path, target_imgs_path, target_labels_path)