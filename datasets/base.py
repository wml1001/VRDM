from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import numpy as np
import yaml
import cv2
import torch

# from SwinTransformerMini import model
from skimage.metrics import structural_similarity as ssim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ImagePathDataset(Dataset):
    def __init__(self, image_paths, image_size=(64, 64), flip=False, to_normal=False):
        self.image_size = image_size
        self.image_paths = image_paths
        self._length = len(image_paths)
        self.flip = flip
        self.to_normal = to_normal # 是否归一化到[-1, 1]

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = 0.0
        if index >= self._length:
            index = index - self._length
            p = 1.0

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])
         # 图像预处理
        transform1 = transforms.Compose([
            # transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])


        img_path = self.image_paths[index]
        
        image = None


        image = Image.open(img_path)
        noise_matrix = np.random.randn(64, 64)
        img_array = np.array(image)
        img_array1 = noise_matrix + img_array
        temp = int(ssim(img_array, img_array1, data_range=255, multichannel=False)*10000000) 
        temp = (temp -  9892313)/(9998405 - 9892313)#test
        # temp = (temp -  9862823)/(9998475 - 9862823)#train
    
        step = int(100 + temp*100)

        # 对图像进行预处理
        # img = transform1(image).unsqueeze(0)  # 增加 batch 维度
        # # 将图像移动到设备上
        # # img = img.to(device)

        # # 使用模型进行预测
        # with torch.no_grad():
        #     output = model(img)

        # # 将输出通过softmax转换为概率分布
        # prob_dist = torch.nn.functional.softmax(output, dim=1)

        # # 找到概率最大的类别下标
        # _, step = torch.max(prob_dist, 1)
        # step = 200

        
        # img = img_read(image)
        # step = pixel_difference_sum(img)

        # try:
        #     image = Image.open(img_path)
        #     # img = self.img_read(image)
        #     # step = self.pixel_difference_sum(img)

        # except BaseException as e:
        #     print(img_path)

        # u, _ = mean_std(np.array(image))
        # u_img = np.full_like(np.array(image), u)  # 创建一个与原始图像相同形状的数组，填充值为均值
        # step = get_mse(np.array(image), u_img)
        # step = 1281 - step/6200
        
        image = transform(image)

        if self.to_normal:
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)

        image_name = Path(img_path).stem
        return image, image_name,step

