import argparse

import warnings
warnings.filterwarnings('ignore')

from exp import Exp

def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='Debug', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--seed', default=1, type=int)

    # dataset parameters
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--val_batch_size', default=16, type=int, help='Batch size')

    # parser.add_argument('--data_root', default='../data/')
    parser.add_argument('--data_root', default='/4TB/hcmeng/Hos_pj/Data')
    parser.add_argument('--num_workers', default=8, type=int)

    # Training parameters
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--dice_w', default=10.0, type=float, help='Dice Loss weight')
    parser.add_argument('--drop_r', default=0.4, type=float, help='drop out rate')
    return parser

if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__

    exp = Exp(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>  start <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.train(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>> testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    mse = exp.test(args)
