from .transformer import ViViViT
from .unet import UNet3D


def build_model(config):
    if config['model_type'] == 'vivit_overlap':
        return ViViViT(config)
    if config['model_type'] == 'unet':
        return UNet3D(base_channels=config['base_channels'], num_levels=config['num_levels'], use_checkpoint_cnn=config['use_checkpoint_cnn'])
    raise ValueError(config['model_type'])
