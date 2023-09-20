from options.base_options import BaseOptions
from os import makedirs
import numpy as np
import torch
import cv2
from rembg import remove
import torch.nn.functional as F
from pre_dataset import *
import warnings
import time
from openvino.runtime import Core
warnings.filterwarnings("ignore")


# Example: python Try_on.py --image_path ./dataset/test_img/000010_0.jpg --clothe_path ./dataset/test_clothes/000010_1.jpg

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
        output = remove(image)
        white_image = Image.new('RGBA', output.size, (255, 255, 255, 255))
        result = Image.alpha_composite(white_image, output)
        image = result.convert('RGB')

        image = image.resize((192, 256),Image.ANTIALIAS)
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

device = torch.device("cpu")
str_compile_time = time.time()
ie = Core()
warp_model_xml = "checkpoints/openvino/IR/Warp_Model.xml"
warp_model_bin = "checkpoints/openvino/IR/Warp_Model.bin"
warpmodel = ie.read_model(model=warp_model_xml, weights=warp_model_bin)
warp_model = ie.compile_model(model=warpmodel, device_name="CPU")
warpout_layer_0 = warp_model.output(0)
warpout_layer_1 = warp_model.output(1)

gen_model_xml = "checkpoints/openvino/IR/Gen_Model.xml"
gen_model_bin = "checkpoints/openvino/IR/Gen_Model.bin"
genmodel = ie.read_model(model=gen_model_xml, weights=gen_model_bin)
gen_model = ie.compile_model(model=genmodel, device_name="CPU")
genout_layer = gen_model.output(0)
end_compile_time = time.time()

print("Start try on project")
str_tryon_time = time.time()
opt = BaseOptions().parse()
data_loader = CustomDataLoader(opt)
dataset = data_loader.load_data()
num = len(dataset)

makedirs(opt.save_path, exist_ok=True)
print("trying on...")

for i, data in enumerate(dataset, 0):
    real_image = data['image']
    clothes = data['clothes']
    edge = data['edge']
    edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int32))
    clothes = clothes * edge

    # for multiple inputs in a list
    Model_layer = warp_model([real_image.to(device), clothes.to(device)])
    warped_cloth_1 = Model_layer[warpout_layer_0]
    last_flow_1 = Model_layer[warpout_layer_1]
    warped_cloth = torch.from_numpy(warped_cloth_1)
    last_flow = torch.from_numpy(last_flow_1)

    warped_edge = F.grid_sample(edge.to(device), last_flow.permute(0, 2, 3, 1),
                      mode='bilinear', padding_mode='zeros')

    gen_inputs = torch.cat([real_image.to(device), warped_cloth.to(device), warped_edge.to(device)], 1)
    gen_outputs = torch.from_numpy(gen_model(gen_inputs)[genout_layer])
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

    FilePath = 'result/Tryon'
    
    os.makedirs(FilePath, exist_ok=True)
    cv2.imwrite(FilePath + '/' + str(i) + '.jpg', cv_img)
end_tryon_time = time.time()
end_time = time.time()
print("Load and compilt model, used {} seconds, calculate fps={}".format(end_compile_time-str_compile_time, num/(end_compile_time-str_compile_time)))
print("Tryon model, used {} seconds, calculate fps={}".format(end_tryon_time-str_tryon_time, num/(end_tryon_time-str_tryon_time)))
print("Successfully tried on the clothes, used {} seconds, calculate fps={}".format(end_time-start_time, num/(end_time-start_time)))

# 计算FID指标
# way1: python -m pytorch_fid dataset/images result/Tryon 
# way2: fidelity --gpu 0 --fid --input1 results/demo/ --input2 dataset/test_img/