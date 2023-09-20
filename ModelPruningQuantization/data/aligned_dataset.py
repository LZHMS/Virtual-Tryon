import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch
import json
import numpy as np
import os.path as osp
from PIL import ImageDraw


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.diction={}
        self.fine_height=256
        self.fine_width=192
        self.radius=5

        # 加载训练数据
        if opt.isTrain or opt.use_encoded_image:
            dir_A = '_A' if self.opt.label_nc == 0 else '_label'
            self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
            self.A_paths = sorted(make_dataset(self.dir_A))

            dir_B = '_B' if self.opt.label_nc == 0 else '_img'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
            self.B_paths = sorted(make_dataset(self.dir_B))

            dir_E = '_edge'
            self.dir_E = os.path.join(opt.dataroot, opt.phase + dir_E)
            self.E_paths = sorted(make_dataset(self.dir_E))

            dir_C = '_color'
            self.dir_C = os.path.join(opt.dataroot, opt.phase + dir_C)
            self.C_paths = sorted(make_dataset(self.dir_C))

        self.dataset_size = len(self.A_paths)

    # 迭代数据集
    def __getitem__(self, index):
        A_path = self.A_paths[index]
        A = Image.open(A_path).convert('L')

        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('RGB'))
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0

        transform_B = get_transform(self.opt, params)
        B_path = self.B_paths[index]
        B = Image.open(B_path).convert('RGB')
        B_tensor = transform_B(B)

        C_path = self.C_paths[index]
        C = Image.open(C_path).convert('RGB')
        C_tensor = transform_B(C)

        E_path = self.E_paths[index]
        E = Image.open(E_path).convert('L')
        E_tensor = transform_A(E)

        index_un = np.random.randint(14221)
        C_un_path = self.C_paths[index_un]
        C_un = Image.open(C_un_path).convert('RGB')
        C_un_tensor = transform_B(C_un)

        E_un_path = self.E_paths[index_un]
        E_un = Image.open(E_un_path).convert('L')
        E_un_tensor = transform_A(E_un)

        pose_name = B_path.replace('.png', '_keypoints.json').replace('.jpg','_keypoints.json').replace('train_img', 'train_pose')
        with open(osp.join(pose_name), 'r') as f:
            pose_label = json.load(f)
            try:
                pose_data = pose_label['people'][0]['pose_keypoints']
            except IndexError:
                pose_data = [0 for i in range(54)]
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))

        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
        r = self.radius
        im_pose = Image.new('L', (self.fine_width, self.fine_height))
        pose_draw = ImageDraw.Draw(im_pose)
        for i in range(point_num):
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            point_x = pose_data[i, 0]
            point_y = pose_data[i, 1]
            if point_x > 1 and point_y > 1:
                draw.rectangle((point_x - r, point_y - r, point_x + r, point_y + r), 'white', 'white')
                pose_draw.rectangle((point_x - r, point_y - r, point_x + r, point_y + r), 'white', 'white')
            one_map = transform_B(one_map.convert('RGB'))
            pose_map[i] = one_map[0]
        P_tensor = pose_map

        dense_pose_name = B_path.replace('.png', '.npy').replace('.jpg', '.npy').replace('train_img', 'train_densepose')
        dense_mask = np.load(dense_pose_name).astype(np.float32)
        dense_mask = transform_A(dense_mask)

        if self.opt.isTrain:
            input_dic = {'label': A_tensor, 'image': B_tensor, 'path': A_path, 'img_path': B_path, 'color_path': C_path,
                'color_un_path': C_un_path, 'edge': E_tensor, 'color': C_tensor, 'edge_un': E_un_tensor,
                'color_un': C_un_tensor, 'pose': P_tensor, 'densepose': dense_mask}
            return input_dic

    def __len__(self):
        return len(self.A_paths) // (self.opt.batchSize * self.opt.num_gpus) * (self.opt.batchSize * self.opt.num_gpus)

    def name(self):
        return 'AlignedDataset'
