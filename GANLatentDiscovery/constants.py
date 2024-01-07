from GANLatentDiscovery.latent_deformator import DeformatorType
from GANLatentDiscovery.trainer import ShiftDistribution


HUMAN_ANNOTATION_FILE = 'human_annotation.txt'


DEFORMATOR_TYPE_DICT = {
    'fc': DeformatorType.FC,
    'linear': DeformatorType.LINEAR,
    'id': DeformatorType.ID,
    'ortho': DeformatorType.ORTHO,
    'proj': DeformatorType.PROJECTIVE,
    'random': DeformatorType.RANDOM,
}


SHIFT_DISTRIDUTION_DICT = {
    'normal': ShiftDistribution.NORMAL,
    'uniform': ShiftDistribution.UNIFORM,
    None: None
}


WEIGHTS = {
    'BigGAN': 'gan_models/pretrained/generators/BigGAN/G_ema.pth',
    'ProgGAN': 'gan_models/pretrained/generators/ProgGAN/100_celeb_hq_network-snapshot-010403.pth',
    'SN_MNIST': 'gan_models/pretrained/generators/SN_MNIST',
    'SN_Anime': 'gan_models/pretrained/generators/SN_Anime',
    'StyleGAN2': 'gan_models/pretrained/StyleGAN2/stylegan2-car-config-f.pt',
}
