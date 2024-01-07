## Under review at CVPR 2024

## Patch of Invisibility: Naturalistic Physical Black-Box Adversarial Attacks on Object Detectors

### Abstract
Adversarial attacks on deep-learning models have been receiving increased attention in recent years. Work in this area has mostly focused on gradient-based techniques, so-called ``white-box'' attacks, wherein the attacker has access to the targeted model's internal parameters; such an assumption is usually unrealistic in the real world. Some attacks additionally use the entire pixel space to fool a given model, which is neither practical nor physical (i.e., real-world). On the contrary, we propose herein a direct, black-box, gradient-free method that uses the learned image manifold of a pretrained generative adversarial network (GAN) to generate naturalistic physical adversarial patches for object detectors. To our knowledge this is the first and only method that performs black-box physical attacks **directly** on object-detection models, which results with a model-agnostic attack. We show that our proposed method works both digitally and physically. We compared our approach against three models and four black-box attacks with different configurations. Our approach outperformed **all** other approaches that were tested in our experiments by a large margin.


## News
- **Jan 7, 2024**: Open source

## Installation
### Clone the code and build the environment
Clone the code:
```bash
git clone https://github.com/razla/Patch-Of-Invisibility.git
cd Patch-Of-Invisibility
```
Build the environment and install PyTorch and Torchvision as following [official PyTorch instruction](https://pytorch.org/get-started/locally/)
```bash
conda create -n advpatch python=3.7
conda activate advpatch
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
```

Install other packages using the following command:
```bash
pip install -r requirements.txt
```
### Dataset
Download the INRIA dataset using following command:
```bash
bash download_inria.sh
```
The original INRIA dataset comes from [INRIA](http://pascal.inrialpes.fr/data/human/).

Check the dataset position:
```
Naturalistic-Adversarial-Patch                           
 └─── dataset
        └───── inria
                └───── Test
                        └───── ...
                └───── Train
                        └───── pos
                                └───── yolo-labels_yolov4tiny
                                └───── *.png
                                └───── ...
 
```

### Pretrained weights  
The proposed method needs a GAN and a detector to generate an adversarial patch. 
You can download the necessary weights by running the following command:
- BigGAN:
```bash
bash ./GANLatentDiscovery/download_weights.sh
```
- YOLOv4 and YOLOv4tiny:
```bash 
bash ./pytorchYOLOv4/weight/download_weights.sh
```
- YOLOv3 and YOLOv3tiny:
```bash 
bash ./PyTorchYOLOv3/weights/download_weights.sh
```
- YOLOv2:
```bash 
bash ./adversarialYolo/weights/download_weight.sh
```
## How to Run
After you prepare the weights and dataset, you can evaluate or generate a naturalistic adversarial patch:
### Test an adversarial patch:
```bash
CUDA_VISIBLE_DEVICES=0 python evaluation.py --model yolov4 --tiny --patch ./patch_sample/v4tiny.png
```
- `--model`: detector model. You can use yolov2, yolov3, yolov4, or fasterrcnn.
- `--tiny`: only works for YOLOv3 and YOLOv4. To use TOLOv4tiny, enable this argument.
- `--patch`: the patch position. 

### Train an adversarial patch:
To train an adversarial patch using YOLOv4tiny:
```bash
CUDA_VISIBLE_DEVICES=0 python ensemble.py --model=yolov4 --tiny
```
- `--model`: detector model. You can use yolov2, yolov3, yolov4, or fasterrcnn.
- `--tiny`: only works for YOLOv3 and YOLOv4. To use TOLOv4tiny, enable this argument.
- `--classBiggan`: the class of generated patch. You can choose from 0 to 999 (ImageNet pretrained). 

The result (i.e, adversarial patch) will be saved at exp/exp{experiemnt id} automatically.
You can use tensorboard to check the training history: 
```bash
tensorboard --logdir=./exp 
```

## Credits
- BigGAN code and weights are base on: [GANLatentDiscovery](https://github.com/anvoynov/GANLatentDiscovery)

- StyleGAN2 code and wieghts are based on: [stylegan2](https://github.com/NVlabs/stylegan2)

- YOLOv2 and adversarial patch codes are based on: [adversarial-yolo](https://gitlab.com/EAVISE/adversarial-yolo)

- YOLOv3 code and weights are based on: [PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)

- YOLOv4 code and weights are based on: [pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4)
## Citation

```
@inproceedings{hu2021naturalistic,
  title={Naturalistic Physical Adversarial Patch for Object Detectors},
  author={Hu, Yu-Chih-Tuan and Kung, Bo-Han and Tan, Daniel Stanley and Chen, Jun-Cheng and Hua, Kai-Lung and Cheng, Wen-Huang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}
```
# Patch-Of-Invisibility
