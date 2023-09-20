from PIL import Image
import torchvision.transforms as transforms
import os
import cv2
import numpy as np
import torch

def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)

def get_transform(method=Image.BICUBIC, normalize=True):
    transform_list = [transforms.Lambda(lambda img: __make_power_2(img, float(16), method)), transforms.ToTensor()]
    if normalize:
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transforms.Compose(transform_list)


class CustomDataSet(torch.utils.data.Dataset):
    def __init__(self, opts):
        self.opt = opts
        self.image_path = opts.image_path
        self.clothe_path = opts.clothe_path
        self.images = os.listdir(self.image_path)
        self.clothes = os.listdir(self.clothe_path)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.image_path, self.images[index])).convert('RGB')
        origin_clothe = Image.open(os.path.join(self.clothe_path, self.clothes[index])).convert('RGB')

        transform = get_transform()
        transform_E = get_transform(method=Image.NEAREST, normalize=False)

        image = transform(image)
        clothe = transform(origin_clothe)
        # get edge
        max_limit = 250
        Edge = cv2.cvtColor(np.array(origin_clothe), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(Edge, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # 二值化——强分离
        blur[blur < max_limit] = 0
        clothe_edge = cv2.cvtColor(255 - blur, cv2.COLOR_BGR2RGB)
        clothe_edge = Image.fromarray(clothe_edge)
        clothe_edge = transform_E(clothe_edge.convert('L'))

        return (image, clothe, clothe_edge), image

    def __len__(self):
        return len(self.images)