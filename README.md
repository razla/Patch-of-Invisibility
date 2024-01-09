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
Build the environment and install the packages
```bash
conda create -n advpatch python=3.10
conda activate advpatch
pip install -r requirements.txt
```

### Dataset
Download the INRIA dataset using following command:
```bash
bash download_inria.sh
```
The original INRIA dataset comes from [INRIA](http://pascal.inrialpes.fr/data/human/).

Change the dataset path in new_utils.py file.

### Pretrained weights  
The proposed method needs a GAN and a detector to generate an adversarial patch. 
You can download the necessary weights by running the following command:
- BigGAN:
```bash
bash ./GANLatentDiscovery/download_weights.sh
```
- YOLOv5s:
Automatically downloaded
- YOLOv4tiny:
```bash 
bash ./pytorchYOLOv4/weight/download_weights.sh
```
- YOLOv3tiny:
```bash 
bash ./PyTorchYOLOv3/weights/download_weights.sh
```
## How to Run
After you prepare the weights and dataset, you can evaluate or generate a patch:
### Test an adversarial patch:
```bash
python evaluation.py --model yolov4 --tiny --patch ./patch_sample/v4tiny.png
```
- `--model`: detector model. You can use yolov2, yolov3, yolov4, or fasterrcnn.
- `--tiny`: only works for YOLOv3 and YOLOv4. To use TOLOv4tiny, enable this argument.
- `--patch`: the patch position. 

### Train an adversarial patch:
To train an adversarial patch using YOLOv4tiny:
```bash
python ensemble.py --model=yolov4 --tiny --method=bbgan --epochs=1000
```
- `--model`: detector model. You can use yolov3, yolov4, or yolov5s.
- `--tiny`: only works for YOLOv3 and YOLOv4. To use TOLOv4tiny, enable this argument.
- `--method`: which kind of method to use (bbgan/raw/random_raw/random_gan/nes).
- `--epochs`: number of epochs to train.

If you want to restore the experiments in the paper, you have all the hyperparameters used in `records.md`

## Credits
- BigGAN code and weights are base on: [GANLatentDiscovery](https://github.com/anvoynov/GANLatentDiscovery)

- YOLOv3 code and weights are based on: [PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)

- YOLOv4 code and weights are based on: [PyTorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4)

- YOLOv5 code and weights are based on: [PyTorch-YOLOv5](https://github.com/ultralytics/yolov5)
## Citation

```
@article{lapid2023patch,
  title={Patch of invisibility: Naturalistic black-box adversarial attacks on object detectors},
  author={Lapid, Raz and Sipper, Moshe},
  journal={arXiv preprint arXiv:2303.04238},
  year={2023}
}
```
