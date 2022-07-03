import configargparse

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
parser.add_argument('--crop-size', type=int, default=256, help='Size of the input (squared) patches')
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
parser.add_argument('--gradient-clip', type=float, default=0.01, help='If > 0, clips gradient norm to that value')

# model
parser.add_argument('--feature-extractor', type=str, default='UResNet', help='Feature extractor for edge potentials')
parser.add_argument('--lambda-init', type=float, default=1., help='Graph lambda parameter initialization')
parser.add_argument('--mu-init', type=float, default=.1, help='Graph mu parameter initialization')
