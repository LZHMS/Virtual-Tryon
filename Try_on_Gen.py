from options.base_options import BaseOptions
from networks import ResUnetGenerator
from afwm import AFWM, IFPN, CFPN
import os
import numpy as np
import torch.nn.functional as F
import torch
import cv2
from PIL import Image
import torch_pruning as tp
from pre_dataset import *
import time
import warnings
warnings.filterwarnings("ignore")

# Example: python test2.py --image_path ./dataset/test_img/000010_0.jpg --clothe_path ./dataset/test_clothes/000010_1.jpg
# ./TryOn --image_path ./dataset/test_img/000010_0.jpg --clothe_path ./dataset/test_clothes/000010_1.jpg

start_time = time.time()
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

        return {'image': image, 'clothes': clothe, 'edge': clothe_edge}

    def __len__(self):
        return len(self.images)


class CustomDataLoader:
    def __init__(self, opts):
        self.dataset = CustomDataSet(opts)
        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=0)

    def load_data(self):
        return self.data_loader

str_dataset_time = time.time()
opt = BaseOptions().parse()

# create result path
FilePath = os.path.join(opt.save_path, opt.name)
os.makedirs(FilePath, exist_ok=True)
data_loader = CustomDataLoader(opt)
dataset = data_loader.load_data()
num = len(dataset)
print("Number of the images:", num)
end_dataset_time = time.time()

print("Start loading models...")
str_compile_time = time.time()
device = torch.device("cpu")
warp_model = AFWM(opt, 3)
warp_model.eval()
#---------------------------Create Prepruning Subnetwork-------------------
IFPN_Module = IFPN()
CFPN_Module = CFPN(3)
IFPN_Module.eval()
CFPN_Module.eval()

IFPN_DG = tp.DependencyGraph().build_dependency(IFPN_Module, torch.randn(1, 3, 256, 192))
CFPN_DG = tp.DependencyGraph().build_dependency(CFPN_Module, torch.randn(1, 3, 256, 192))
warp_state_dict = torch.load(opt.warp_checkpoint, map_location=torch.device('cpu'))
IFPN_DG.load_pruning_history(warp_state_dict['IFPN_pruning'])
CFPN_DG.load_pruning_history(warp_state_dict['CFPN_pruning'])
warp_model.image_features, warp_model.cond_features = IFPN_Module.image_features, CFPN_Module.cond_features
warp_model.image_FPN, warp_model.cond_FPN = IFPN_Module.image_FPN, CFPN_Module.cond_FPN
# warp_state_dict['model'].pop('aflow_net.weight')
warp_model.load_state_dict(warp_state_dict['model'])
warp_model.to(device)
print("This is the warp_model structure:\n", warp_model)

gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=torch.nn.BatchNorm2d)
gen_model.eval()
DG = tp.DependencyGraph().build_dependency(gen_model, torch.randn(1, 7, 256, 192))
gen_state_dict = torch.load(opt.gen_checkpoint, map_location=torch.device('cpu'))
DG.load_pruning_history(gen_state_dict['pruning'])
gen_model.load_state_dict(gen_state_dict['model'])
gen_model.to(device)

print("This is the gen_model structure:\n", gen_model)
end_compile_time = time.time()

str_tryon_time = time.time()
for i, data in enumerate(dataset, 0):
    real_image = data['image']
    clothes = data['clothes']
    edge = data['edge']
    edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int))
    clothes = clothes * edge
    
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
    
    cv2.imwrite(FilePath + '/' + str(i) + '.jpg', cv_img)

end_tryon_time = time.time()
end_time = time.time()
print("Load and compilt model, used {} seconds, calculate fps={}".format(end_compile_time-str_compile_time, 1/(end_compile_time-str_compile_time)))
# print("Load dataset, used {} seconds, calculate fps={}".format(end_dataset_time-str_dataset_time, 1/(end_dataset_time-str_dataset_time)))
print("Tryon model, used {} seconds, calculate fps={}".format(end_tryon_time-str_tryon_time, num/(end_tryon_time-str_tryon_time)))
print("Successfully tried on the clothes, used {} seconds, calculate fps={}".format(end_time-start_time, num/(end_time-start_time)))

# Calculate FID
# python -m pytorch_fid result/Try_on_Gen_0.2 dataset/images
