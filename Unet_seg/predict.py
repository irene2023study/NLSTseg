import os
from PIL import Image
import numpy as np
import torch

from diffusion_model import UNet
from torchvision import transforms

Patient_path = '203166'
CT_path = os.path.join(Patient_path, 'ct')
Tumor_path = os.path.join(Patient_path, 'predict')
if not os.path.exists(Tumor_path):
    os.makedirs(Tumor_path)


transform = transforms.Compose([
                    # transforms.Resize([512,512]) , 
                    transforms.ToTensor(),
                    # transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                    transforms.Normalize(mean = [0.5], std = [0.5])
                ])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = UNet(
                img_channels = 1,
                base_channels=64,
                channel_mults=(1, 2),
                time_dim=256,
                num_res_blocks=2,
                dropout= 0.4
                )
model = model.to(device)
new_weight = model.state_dict()
training_state = torch.load('./results/D2_oversamp_e200_lr1e4_Dice10_unet3/checkpoint.pth', map_location=device)
new_weight.update(training_state)
new_weight = {key.replace('module.', ''): value for key, value in new_weight.items()}
model.load_state_dict(new_weight)
model.eval()
for i in os.listdir(CT_path):
    img = Image.open(os.path.join(CT_path, i))
    img = transform(img).to(device)

    img = torch.unsqueeze(img, dim=0)
    seg = model(img, None)
    seg = torch.sigmoid(seg)
    seg = (seg > 0.5).float()  # 根據閾值進行二值化
    # 將 PyTorch tensor 轉換為 NumPy array
    seg_np = seg.squeeze().cpu().numpy()
    # 將數值範圍歸一化到 [0, 255]
    seg_np = (seg_np * 255).astype(np.uint8)

    # 將 NumPy array 轉換為 Pillow 的 Image
    seg_img = Image.fromarray(seg_np)

    # 保存圖像
    seg_img.save(os.path.join(Tumor_path, i))