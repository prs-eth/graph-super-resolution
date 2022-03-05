import os
import argparse
from collections import defaultdict
import time

import configargparse
import numpy as np
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from tqdm import tqdm

from model import GraphSuperResolutionNet
from data import MiddleburyDataset, NYUv2Dataset, DIMLDataset
from utils import new_log, to_cuda, seed_all

parser = configargparse.ArgumentParser()
parser.add_argument('-c', '--config', is_config_file=True, help='Path to the config file', type=str)

# general
parser.add_argument('--save-dir', required=True, help='Path to directory where models and logs should be saved')
parser.add_argument('--logstep-train', default=10, type=int, help='Training log interval in steps')
parser.add_argument('--save-model', default='both', choices=['last', 'best', 'no', 'both'])
parser.add_argument('--val-every-n-epochs', type=int, default=1, help='Validation interval in epochs')
parser.add_argument('--resume', type=str, default=None, help='Checkpoint path to resume')
parser.add_argument('--seed', type=int, default=12345, help='Random seed')
parser.add_argument('--wandb', action='store_true', default=False, help='Use Weights & Biases instead of TensorBoard')
parser.add_argument('--wandb-project', type=str, default='graph-sr', help='Wandb project name')

# data
parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
parser.add_argument('--data-dir', type=str, required=True, help='Root directory of the dataset')
parser.add_argument('--num-workers', type=int, default=8, metavar='N', help='Number of dataloader worker processes')
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--crop_size', type=int, default=256, help='Size of the input (squared) patches')
parser.add_argument('--scaling', type=int, default=8, help='Scaling factor')
parser.add_argument('--max-rotation', type=float, default=15., help='Maximum rotation angle (degrees)')
parser.add_argument('--no-flip', action='store_true', default=False, help='Switch off random flipping')
parser.add_argument('--in-memory', action='store_true', default=False, help='Hold data in memory during training')

# training
parser.add_argument('--loss', default='l1', type=str, choices=['l1', 'mse'])
parser.add_argument('--num-epochs', type=int, default=250)
parser.add_argument('--optimizer', default='adam', choices=['sgd', 'adam'])
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--w-decay', type=float, default=1e-5)
parser.add_argument('--lr-scheduler', type=str, default='step', choices=['no', 'step', 'plateau'])
parser.add_argument('--lr-step', type=int, default=10, help='LR scheduler step size (epochs)')
parser.add_argument('--lr-gamma', type=float, default=0.9, help='LR decay rate')
parser.add_argument('--skip-first', action='store_true', help='Don\'t optimize during first epoch')
parser.add_argument('--gradient-clip', type=float, default=0., help='If > 0, clips gradient norm to that value')

# model
parser.add_argument('--feature-extractor', type=str, default='UResNet', help='Feature extractor for edge potentials')
parser.add_argument('--pretrained', action='store_true', help='Initialize feature extractor with weights '
                                                              'pretrained on ImageNet')
parser.add_argument('--lambda-init', type=float, default=1., help='Graph lambda parameter initialization')
parser.add_argument('--mu-init', type=float, default=.1, help='Graph mu parameter initialization')


class Trainer:

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.use_wandb = self.args.wandb

        self.dataloaders = self.get_dataloaders(args)
        
        seed_all(args.seed)

        self.model = GraphSuperResolutionNet(
            args.scaling,
            args.crop_size,
            args.feature_extractor,
            args.pretrained,
            args.lambda_init,
            args.mu_init
        )
        if args.resume is not None:
            self.resume(path=args.resume)
        self.model.cuda()

        self.experiment_folder = new_log(os.path.join(args.save_dir, args.dataset), args)

        if self.use_wandb:
            wandb.init(project=args.wandb_project, dir=self.experiment_folder)
            wandb.config.update(self.args)
            self.writer = None
        else:
            self.writer = SummaryWriter(log_dir=self.experiment_folder)

        if args.optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.w_decay)
        elif args.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=self.args.momentum,
                                       weight_decay=args.w_decay)

        if args.lr_scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
        elif args.lr_scheduler == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=args.lr_step,
                                                                  factor=args.lr_gamma)
        else:
            self.scheduler = None

        self.epoch = 0
        self.iter = 0
        self.train_stats = defaultdict(lambda: np.nan)
        self.val_stats = defaultdict(lambda: np.nan)
        self.best_optimization_loss = np.inf

    def __del__(self):
        if not self.use_wandb:
            self.writer.close()

    def train(self):
        with tqdm(range(0, self.args.num_epochs), leave=True) as tnr:
            tnr.set_postfix(training_loss=np.nan, validation_loss=np.nan, best_validation_loss=np.nan)
            for _ in tnr:
                self.train_epoch(tnr)

                if (self.epoch + 1) % self.args.val_every_n_epochs == 0:
                    self.validate()

                    if self.args.save_model in ['last', 'both']:
                        self.save_model('last')

                if self.args.lr_scheduler == 'step':
                    self.scheduler.step()
                    if self.use_wandb:
                        wandb.log({'log_lr': np.log10(self.scheduler.get_last_lr())}, self.iter)
                    else:
                        self.writer.add_scalar('log_lr', np.log10(self.scheduler.get_last_lr()), self.epoch)

                self.epoch += 1

    def train_epoch(self, tnr=None):
        self.train_stats = defaultdict(float)

        self.model.train()

        with tqdm(self.dataloaders['train'], leave=False) as inner_tnr:
            inner_tnr.set_postfix(training_loss=np.nan)
            for i, sample in enumerate(inner_tnr):
                sample = to_cuda(sample)

                self.optimizer.zero_grad()

                output = self.model(sample)

                loss, loss_dict = self.model.get_loss(output, sample, kind=self.args.loss)

                for key in loss_dict:
                    self.train_stats[key] += loss_dict[key]

                if self.epoch > 0 or not self.args.skip_first:
                    loss.backward()

                    if self.args.gradient_clip > 0.:
                        clip_grad_norm_(self.model.parameters(), self.args.gradient_clip)

                    self.optimizer.step()

                self.iter += 1

                if (i + 1) % min(self.args.logstep_train, len(self.dataloaders['train'])) == 0:
                    self.train_stats = {k: v / self.args.logstep_train for k, v in self.train_stats.items()}

                    inner_tnr.set_postfix(training_loss=self.train_stats['optimization_loss'])
                    if tnr is not None:
                        tnr.set_postfix(training_loss=self.train_stats['optimization_loss'],
                                        validation_loss=self.val_stats['optimization_loss'],
                                        best_validation_loss=self.best_optimization_loss)

                    if self.use_wandb:
                        wandb.log({k + '/train': v for k, v in self.train_stats.items()}, self.iter)
                    else:
                        for key in self.train_stats:
                            self.writer.add_scalar('train/' + key, self.train_stats[key], self.iter)

                    # reset metrics
                    self.train_stats = defaultdict(float)

    def validate(self):
        self.val_stats = defaultdict(float)

        self.model.eval()

        with torch.no_grad():
            for sample in tqdm(self.dataloaders['val'], leave=False):
                sample = to_cuda(sample)

                output = self.model(sample)

                loss, loss_dict = self.model.get_loss(output, sample, kind=self.args.loss)

                for key in loss_dict:
                    self.val_stats[key] += loss_dict[key]

            self.val_stats = {k: v / len(self.dataloaders['val']) for k, v in self.val_stats.items()}

            if self.use_wandb:
                wandb.log({k + '/val': v for k, v in self.val_stats.items()}, self.iter)
            else:
                for key in self.val_stats:
                    self.writer.add_scalar('val/' + key, self.val_stats[key], self.epoch)

            if self.val_stats['optimization_loss'] < self.best_optimization_loss:
                self.best_optimization_loss = self.val_stats['optimization_loss']
                if self.args.save_model in ['best', 'both']:
                    self.save_model('best')

    @staticmethod
    def get_dataloaders(args):
        data_args = {
            'crop_size': (args.crop_size, args.crop_size),
            'in_memory': args.in_memory,
            'max_rotation_angle': args.max_rotation,
            'do_horizontal_flip': not args.no_flip,
            'crop_valid': True,
            'image_transform': Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            'scaling': args.scaling
        }

        phases = ('train', 'val')
        if args.dataset == 'Middlebury':
            depth_transform = Normalize([2296.78], [1122.7])
            datasets = {phase: MiddleburyDataset(os.path.join(args.data_dir, 'Middlebury'), **data_args, split=phase,
                        depth_transform=depth_transform, crop_deterministic=phase == 'val') for phase in phases}

        elif args.dataset == 'DIML':
            depth_transform = Normalize([2749.64], [1154.29])
            datasets = {phase: DIMLDataset(os.path.join(args.data_dir, 'DIML'), **data_args, split=phase,
                        depth_transform=depth_transform) for phase in phases}

        elif args.dataset == 'NYUv2':
            depth_transform = Normalize([2796.32], [1386.05])
            datasets = {phase: NYUv2Dataset(os.path.join(args.data_dir, 'NYU Depth v2'), **data_args, split=phase,
                        depth_transform=depth_transform) for phase in phases}
        else:
            raise NotImplementedError(f'Dataset {args.dataset}')

        return {phase: DataLoader(datasets[phase], batch_size=args.batch_size, num_workers=args.num_workers,
                shuffle=True, drop_last=False) for phase in phases}

    def save_model(self, prefix=''):
        torch.save(self.model.state_dict(), os.path.join(self.experiment_folder, f'{prefix}_model.pth'))

    def resume(self, path):
        if not os.path.isfile(path):
            raise RuntimeError(f'No checkpoint found at \'{path}\'')
        self.model.load_state_dict(torch.load(path))
        print(f'Checkpoint \'{path}\' loaded.')


if __name__ == '__main__':
    args = parser.parse_args()
    print(parser.format_values())

    if args.wandb:
        import wandb

    trainer = Trainer(args)

    since = time.time()
    trainer.train()
    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
