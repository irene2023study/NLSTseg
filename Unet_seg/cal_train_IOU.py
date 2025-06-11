
import os
import torch
import torch.nn as nn

from API.metrics import calculate_iou
from dataloader import Lung_CT
from diffusion_model import UNet
from tqdm import tqdm

train_set = Lung_CT(root= '/4TB/hcmeng/Hos_pj/Data', mode='test', oversam = False)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=8, shuffle=False, pin_memory=True, num_workers=8)

# os.environ['CUDA_VISIBLE_DEVICES'] = '8'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(
                img_channels = 1,
                base_channels=64,
                channel_mults=(1, 2),
                time_dim=256,
                num_res_blocks=2,
                dropout= 0.4
                )
model = nn.DataParallel(model)
model = model.to(device)

new_weight = model.state_dict()
training_state = torch.load('/4TB/hcmeng/Hos_pj/Unet_seg/results/D2_oversamp_e200_lr1e4_Dice10_unet3/checkpoint.pth', map_location=device)
new_weight.update(training_state)
# new_weight = {key.replace('module.', ''): value for key, value in new_weight.items()}
model.load_state_dict(new_weight)


model.eval()
train_pbar = tqdm(train_loader)
count = 0
total_iou = 0
for batch_x, batch_y in train_pbar:
    #print(batch_x.shape)
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    # print(torch.max(batch_y))
    pred_y = model(batch_x, None)
    iou = calculate_iou((pred_y > 0.5).float(), (batch_y)) 
    if iou is not None:
        total_iou += iou
        count += 1

total_iou /= count
print(total_iou)
   
