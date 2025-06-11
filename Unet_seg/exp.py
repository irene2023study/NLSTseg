
import os
import os.path as osp
import json
import torch
import pickle
import logging
import numpy as np

from tqdm import tqdm
from API.metrics import calculate_iou
# from API.PerceptualLoss import PerceptualLoss, FeatPerceptualLoss
from API.recorder import Recorder
# from unet.unet_model import UNet
from diffusion_model import UNet
from utils import *
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataloader import Lung_CT
from API import *
from loss import DiceLoss
class Exp:
    def __init__(self, args):
        super(Exp, self).__init__()
        self.args = args
        self.config = self.args.__dict__
        self.device = self._acquire_device()

        self._preparation()
        print_log(output_namespace(self.args))
        log_path = '/4TB/hcmeng/Hos_pj/Unet_seg/results/tensorboard_log'
        self.writer = SummaryWriter(log_dir=osp.join(log_path, self.args.ex_name))

    def _acquire_device(self):
        
        if self.args.use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print_log('Use GPU')
        else:
            device = torch.device('cpu')
            print_log('Use CPU')
        return device

    def _preparation(self):
        # seed
        set_seed(self.args.seed)
        # log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')
        # prepare data
        self._get_data()
        # build the model
        self._build_model()

        self._select_optimizer()
        self._select_criterion()

    def _build_model(self):
        args = self.args
        # self.model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        #         in_channels=3, out_channels=1, init_features=32, pretrained=True)
        # self.model = UNet(1, 1, True)
        self.model = UNet(
                img_channels = 1,
                base_channels=64,
                channel_mults=(1, 2),
                time_dim=256,
                num_res_blocks=2,
                dropout=args.drop_r
                )
        
        self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)
        tot = int(sum([np.prod(p.shape) for p in self.model.parameters()]))
        if tot >= 1e6:
            print_log('model: #params={:.1f}M'.format(tot / 1e6))
        else:
            print_log('model: #params={:.1f}K'.format(tot / 1e3))

    def _get_data(self):
        config = self.args.__dict__

        self.train_set = Lung_CT(root= self.args.data_root, mode='train')
        self.vali_set = Lung_CT(root= self.args.data_root, mode='test')

        sampler = WeightedRandomSampler(weights=self.train_set.sample_weights, num_samples=len(self.train_set), replacement=True)

        # # 创建 DataLoader 并使用过采样的 sampler
        self.train_loader = DataLoader(self.train_set, batch_size=self.args.batch_size, sampler=sampler, pin_memory=True, num_workers=self.args.num_workers)

        # self.train_loader = torch.utils.data.DataLoader(
        # self.train_set, batch_size=self.args.batch_size, shuffle=True, pin_memory=True, num_workers=self.args.num_workers)

        self.vali_loader = torch.utils.data.DataLoader(
        self.vali_set, batch_size=self.args.val_batch_size, shuffle=False, pin_memory=True, num_workers=self.args.num_workers)

        # self.vali_loader = self.test_loader if self.vali_loader is None else self.vali_loader
        self.test_loader = self.vali_loader
    def _select_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.args.lr, steps_per_epoch=len(self.train_loader), epochs=self.args.epochs, pct_start=0.1)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer, self.args.epochs*63) 
        return self.optimizer

    def _select_criterion(self):
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.criterion2 = DiceLoss()
        # weight = torch.tensor([4.0]).to(self.device)
        # self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
        # self.criterion = torch.nn.BCELoss()

    def _save(self, epoch, name=''):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, os.path.join(self.checkpoints_path, name + '.pth'))
        
        # torch.save(self.model.state_dict(), os.path.join(
        #     self.checkpoints_path, name + '.pth'))
        state = self.scheduler.state_dict()
        # fw = open(os.path.join(self.checkpoints_path, name + '.pkl'), 'wb')
        # pickle.dump(state, fw)

    def train(self, args):
        config = args.__dict__
        recorder = Recorder(verbose=True)

        for epoch in range(config['epochs']):
            train_loss = []
            dice_loss = []
            self.model.train()
            train_pbar = tqdm(self.train_loader)
            for batch_x, batch_y in train_pbar:
                #print(batch_x.shape)
                self.optimizer.zero_grad()
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                # print(torch.max(batch_y))
                # pred_y = self.model(batch_x)
                pred_y = self.model(batch_x, None)
                loss = self.criterion(pred_y, batch_y) #16 1 512 512
                loss2 = self.criterion2(pred_y, batch_y)

                train_loss.append(loss.item())
                dice_loss.append(loss2.item())
                train_pbar.set_description('train loss: {:.4f} | dice loss: {:.4f}'.format(loss.item(), loss2.item()))

                total_loss = loss + args.dice_w * loss2
                total_loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            train_loss = np.average(train_loss)
            dice_loss = np.average(dice_loss)
            # with torch.no_grad():
            #     _,_, train_iou = self.vali(self.train_loader)

            print_log("Epoch: {0} | Train Loss: {1:.4f} | Train Dice Loss: {2:.4f}\n".format(
                    epoch + 1, train_loss, dice_loss))
            
            if epoch + 1 >= args.log_step:
                with torch.no_grad():
                    vali_loss, vali_dice_loss, vali_iou = self.vali(self.vali_loader)
                    if epoch % (25) == 0:
                        self._save(name=str(epoch), epoch=epoch)
                print_log("Epoch: {0} | Vali Loss: {1:.4f} | Vali Dice Loss: {2:.4f} | Vali iou: {3:.6f}\n".format(
                    epoch + 1, vali_loss, vali_dice_loss, vali_iou))
                self.writer.add_scalar("train/loss", train_loss, epoch)
                self.writer.add_scalar("train/dice_loss", dice_loss, epoch)
                self.writer.add_scalar("vali/loss", vali_loss, epoch)
                self.writer.add_scalar("vali/dice_loss", vali_dice_loss, epoch)
                self.writer.add_scalar("vali/iou", vali_iou, epoch)
                self.writer.add_scalar("train/lr", self.scheduler.get_lr()[0], epoch)
                # self.writer.add_figure("Train Confusion matrix", getConfusionMatrix(preds, trues), epoch)
                if epoch + 1 >= args.log_step:
                    recorder(-vali_iou, self.model, self.path)

        self.writer.close()
        
        # best_model_path = self.path + '/' + 'checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def vali(self, vali_loader):
        self.model.eval()
        total_loss = []
        dice_loss = []
        vali_pbar = tqdm(vali_loader)
        count = 0
        total_iou = 0
        for i, (batch_x, batch_y) in enumerate(vali_pbar):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            # pred_y = self.model(batch_x)
            pred_y = self.model(batch_x, None)

            loss = self.criterion(pred_y, batch_y)
            loss2 = self.criterion2(pred_y, batch_y)
            vali_pbar.set_description(
                'vali loss: {:.4f} | dice loss: {:.4f}'.format(loss.mean().item(), loss2.mean().item()))
            total_loss.append(loss.mean().item())
            dice_loss.append(loss2.mean().item())

            iou = calculate_iou((pred_y > 0.5).float(), (batch_y)) 
            if iou is not None:
                total_iou += iou
                count += 1

        total_loss = np.average(total_loss)
        dice_loss = np.average(dice_loss)

        total_iou /= count

        # print_log('test - iou:{:.6f}'.format(total_iou))

        self.model.train()
        return total_loss, dice_loss, total_iou

    def test(self, args):
        self.model.eval()
        count = 0
        total_iou = 0
        for batch_x, batch_y in self.test_loader:
            batch_y = batch_y.to(self.device)
            #pred_y = self.model(batch_x.to(self.device))
            pred_y = self.model(batch_x.to(self.device), None)

            iou = calculate_iou((pred_y > 0.5).float(), (batch_y)) 
            if iou is not None:
                total_iou += iou
                count += 1
 

        total_iou /= count
        print_log('test iou:{:.6f}'.format(total_iou))

        return total_iou
