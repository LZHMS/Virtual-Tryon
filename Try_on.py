from options.base_options import BaseOptions
from networks import ResUnetGenerator, load_checkpoint
from afwm import AFWM
import os
import numpy as np
import torch
import cv2
import torch.nn.functional as F
from PIL import Image
import torch_pruning as tp
from pre_dataset import *
import warnings
warnings.filterwarnings("ignore")

# Example: python test2.py --image_path ./dataset/test_img/000010_0.jpg --clothe_path ./dataset/test_clothes/000010_1.jpg
# ./TryOn --image_path ./dataset/test_img/000010_0.jpg --clothe_path ./dataset/test_clothes/000010_1.jpg


class CustomDataSet(torch.utils.data.Dataset):
    def __init__(self, opts):
        self.opt = opts
        self.image_path = opts.image_path
        self.clothes_path = opts.clothe_path

    def __getitem__(self, index):
        image = Image.open(self.image_path).convert('RGB')
        origin_clothe = Image.open(self.clothes_path).convert('RGB')

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

        return {'image': image, 'clothes': clothe, 'edge': clothe_edge}

    def __len__(self):
        return 1


class CustomDataLoader:
    def __init__(self, opts):
        self.dataset = CustomDataSet(opts)
        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=0)

    def load_data(self):
        return self.data_loader


opt = BaseOptions().parse()

data_loader = CustomDataLoader(opt)
dataset = data_loader.load_data()
print("Start processing image...")

device = torch.device("cpu")
warp_model = AFWM(opt, 3)
warp_model.eval()
if opt.type == 0:
    load_checkpoint(warp_model, opt.warp_checkpoint)
elif opt.type == 1:
    IF_Module = warp_model.image_features
    DG = tp.DependencyGraph().build_dependency(IF_Module, torch.randn(1, 3, 256, 192))
    state_dict = torch.load(opt.warp_checkpoint)
    DG.load_pruning_history(state_dict['pruning'])
    warp_model.image_features = IF_Module
    warp_model.load_state_dict(state_dict['model'])
warp_model.to(device)


gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=torch.nn.BatchNorm2d)
gen_model.eval()
load_checkpoint(gen_model, opt.gen_checkpoint)
gen_model.to(device)

os.makedirs(opt.save_path, exist_ok=True)

for i, data in enumerate(dataset, 0):
    real_image = data['image']
    clothes = data['clothes']
    edge = data['edge']
    edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int))
    clothes = clothes * edge

    print([real_image.shape, clothes.shape])
    warped_cloth, last_flow = warp_model(real_image.to(device), clothes.to(device))
    warped_edge = F.grid_sample(edge.to(device), last_flow.permute(0, 2, 3, 1),
                      mode='bilinear', padding_mode='zeros')

    gen_inputs = torch.cat([real_image.to(device), warped_cloth.to(device), warped_edge.to(device)], 1)
    gen_outputs = gen_model(gen_inputs)
    p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
    p_rendered = torch.tanh(p_rendered)
    m_composite = torch.sigmoid(m_composite)
    m_composite = m_composite * warped_edge
    p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

    c = p_tryon
    combine = c.squeeze()
    bgr = (combine.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
    rgb = (bgr * 255).astype(np.uint8)
    cv_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    
    if opt.type == 0:
        FilePath = 'result/Tryon'
    elif opt.type == 1:
        FilePath = 'result/Tryon_IFModule_' + opt.warp_checkpoint[-7:-4]
    
    os.makedirs(FilePath, exist_ok=True)
    cv2.imwrite(FilePath + '/' + str(i) + '.jpg', cv_img)

print("Successfully processed the image!")

# 计算FID指标
# way1: python -m pytorch_fid result/Tryon_IFModule_0.5 dataset/images
# python -m pytorch_fid result/Tryon dataset/images
# way2: fidelity --gpu 0 --fid --input1 results/demo/ --input2 dataset/test_img/
