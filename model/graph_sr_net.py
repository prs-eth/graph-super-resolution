from math import log

import torch
from torch import nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from .functional import create_fixed_cupy_sparse_matrices, GraphQuadraticSolver
from losses import l1_loss_func, mse_loss_func

INPUT_DIM = 4
FEATURE_DIM = 64


def get_neighbor_affinity_no_border(feature_map, mu, lambda_):
    B, M, H, W = feature_map.shape

    feature_map_padded = F.pad(feature_map, (1, 1, 1, 1), 'constant', 0)

    top = torch.mean((feature_map_padded[:, :, 0:-2, 1:-1] - feature_map)**2, dim=1, keepdim=True)
    bottom = torch.mean((feature_map_padded[:, :, 2:, 1:-1] - feature_map)**2, dim=1, keepdim=True)
    left = torch.mean((feature_map_padded[:, :, 1:-1, 0:-2] - feature_map)**2, dim=1, keepdim=True)
    right = torch.mean((feature_map_padded[:, :, 1:-1, 2:] - feature_map)**2, dim=1, keepdim=True)

    affinity = torch.cat([top, bottom, left, right], dim=1) / (1e-6 + mu**2)
    affinity = torch.exp(-affinity)

    border_remover = torch.ones((1, 4, H, W), device=feature_map.device)
    border_remover[0, 0, 0, :] = 0  # top
    border_remover[0, 1, -1, :] = 0  # bottom
    border_remover[0, 2, :, 0] = 0  # left
    border_remover[0, 3, :, -1] = 0  # right

    affinity = affinity * border_remover
    center = torch.sum(affinity, dim=1, keepdim=True)
    affinity = torch.cat([affinity, center], dim=1)
    affinity = affinity * lambda_

    return affinity


class GraphSuperResolutionNet(nn.Module):
    
    def __init__(
            self,
            scaling: int,
            crop_size=256,
            feature_extractor='UResNet',
            pretrained=True,
            lambda_init=1.0,
            mu_init=0.1
    ):
        super().__init__()

        if crop_size not in [64, 128, 256]:
            raise ValueError('Crop size should be in {64, 128, 256}, got ' + str(crop_size))
 
        if feature_extractor == 'Color':
            self.feature_extractor = None
            # so the optimizer does not complain in case we have no other parameters
            self.dummy_param = nn.Parameter(torch.zeros(1))
        elif feature_extractor == 'UResNet':
            self.feature_extractor = smp.Unet('resnet50', classes=FEATURE_DIM, in_channels=INPUT_DIM,
                                              encoder_weights='imagenet' if pretrained else None)
        elif feature_extractor == 'UResNet18':
            self.feature_extractor = smp.Unet('resnet18', classes=FEATURE_DIM, in_channels=INPUT_DIM,
                                              encoder_weights='imagenet' if pretrained else None)
        elif feature_extractor == 'UEffNet2':
            self.feature_extractor = smp.Unet('efficientnet-b2', classes=FEATURE_DIM, in_channels=INPUT_DIM,
                                              encoder_weights='imagenet' if pretrained else None)
        else:
            raise NotImplementedError(f'Feature extractor {feature_extractor}')

        self.log_lambda = nn.Parameter(torch.tensor([log(lambda_init)]))
        self.log_mu = nn.Parameter(torch.tensor([log(mu_init)]))
        self.mx_dict = create_fixed_cupy_sparse_matrices(crop_size, crop_size, scaling)

    def forward(self, sample):
        guide, source, mask_lr = sample['guide'], sample['source'], sample['mask_lr']

        if self.feature_extractor is None:
            pixel_features = torch.cat([guide, sample['y_bicubic']], dim=1)
        else:
            pixel_features = self.feature_extractor(torch.cat([guide, sample['y_bicubic']], dim=1))

        mu, lambda_ = torch.exp(self.log_mu), torch.exp(self.log_lambda)
        neighbor_affinity = get_neighbor_affinity_no_border(pixel_features, mu, lambda_)

        y_pred = GraphQuadraticSolver.apply(neighbor_affinity, source, self.mx_dict, mask_lr)

        return {'y_pred': y_pred, 'neighbor_affinity': neighbor_affinity}

    def get_loss(self, output, sample, kind='l1'):
        y_pred = output['y_pred']
        y, mask_hr, mask_lr = (sample[k] for k in ('y', 'mask_hr', 'mask_lr'))

        l1_loss = l1_loss_func(y_pred, y, mask_hr)
        mse_loss = mse_loss_func(y_pred, y, mask_hr)
        loss = l1_loss if kind == 'l1' else mse_loss

        return loss, {
            'l1_loss': l1_loss.detach().item(),
            'mse_loss': mse_loss.detach().item(),
            'mu': torch.exp(self.log_mu).detach().item(),
            'lambda': torch.exp(self.log_lambda).detach().item(),
            'optimization_loss': loss.detach().item(),
            'average_link': torch.mean(output['neighbor_affinity'][:, 0:4].detach()).item()
        }
