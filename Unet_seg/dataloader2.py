import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import numpy as np

class Lung_CT(data.Dataset):
    def __init__(self, root, mode='train'):
        super(Lung_CT, self).__init__()
        self.mode = mode
        self.imgs = []
        self.labels = []
        txt_file_name = os.path.join(root, mode +'.txt')

        with open(txt_file_name, 'r') as f:
            self.pid_list = f.read().split()
        self.pid_list = sorted(self.pid_list)
        
        for p in self.pid_list:
            imgs_path = os.path.join(root, p, 'ct')
            gts_path = os.path.join(root, p, 'tumor')

            for img in sorted(os.listdir(imgs_path)):
                self.imgs.append(os.path.join(imgs_path, img))
                self.labels.append(os.path.join(gts_path, img))
        
        self.transform = transforms.Compose([
                    # transforms.Resize([512,512]) , 
                    transforms.ToTensor(),
                    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                ])


    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert("RGB")
        img = self.transform(img)
        label = transforms.ToTensor()(Image.open(self.labels[idx]))

        if self.mode == 'testG':
            name = self.imgs[idx].split("/")[-3] +  '_' + self.imgs[idx].split("/")[-1]

            return img, label, name

        return img, label

    def __len__(self):
        return len(self.imgs)
    

dataset1 = Lung_CT(root='/4TB/hcmeng/Hos_pj/Data', mode='train')
print(len(dataset1))
# a = dataset1[5]
# print(torch.max(a[1]))
# print(a)