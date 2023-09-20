#------------------Programe Overview-----------------
# Author: Zhihao Li
# Time: July 20, 2023
# Function: This is used for quantizing the Gen Model using QAT
#------------------Programe Overview-----------------
from options.train_options import TrainOptions
from models.networks import ResUnetGenerator, VGGLoss, load_checkpoint_parallel
from models.afwm import TVLoss, AFWM
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import time
import datetime
import torch
from torch.utils.data import DataLoader
from torch.ao.quantization import (get_default_qconfig_mapping,
                                   get_default_qat_qconfig_mapping,
                                   QConfigMapping)
import torch.ao.quantization.quantize_fx as quantize_fx
import warnings
warnings.filterwarnings("ignore")


def CreateDataset(opt):
    from data.aligned_dataset import AlignedDataset
    dataset = AlignedDataset()
    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


# load operator
opt = TrainOptions().parse()
# create run directory
FilePath = os.path.join(opt.save_checkpoint, opt.name)
os.makedirs(FilePath, exist_ok=True)

torch.cuda.set_device(opt.local_rank)
device = torch.device(f'cuda:{opt.local_rank}')

start_epoch, epoch_iter = 1, 0

train_data = CreateDataset(opt)
train_loader = DataLoader(train_data, batch_size=opt.batchSize, shuffle=False, num_workers=0, pin_memory=True)
dataset_size = len(train_loader)
print('#fine-pruning images = %d' % dataset_size)


#--------------------------Load Pretrained Models-------------------------------
PF_warp_model = AFWM(opt, 3)
PF_warp_model.eval()
PF_warp_model.cuda()
load_checkpoint_parallel(PF_warp_model, opt.PFAFN_warp_checkpoint)

PF_gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=torch.nn.BatchNorm2d)
PF_gen_model.eval()
PF_gen_model.cuda()
load_checkpoint_parallel(PF_gen_model, opt.PFAFN_gen_checkpoint)

PB_warp_model = AFWM(opt, 45)
PB_warp_model.eval()
PB_warp_model.cuda()
load_checkpoint_parallel(PB_warp_model, opt.PBAFN_warp_checkpoint)

PB_gen_model = ResUnetGenerator(8, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
PB_gen_model.eval()
PB_gen_model.cuda()
load_checkpoint_parallel(PB_gen_model, opt.PBAFN_gen_checkpoint)

#PF_warp_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(PF_warp_model).to(device)
#PF_gen_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(PF_gen_model).to(device)

# fuse
try:
    model_fused = quantize_fx.fuse_fx(PF_gen_model)
    print("Successfully fused the gen_model!")
    torch.save(model_fused.state_dict(), os.path.join(FilePath, 'gen_fused_Pytorch.pth'))
except:
    model_fused = PF_gen_model
    print("Can't fuse the gen_model!")

# quantization aware training for static quantization
qconfig_mapping = get_default_qat_qconfig_mapping("qnnpack")
model_fused.train()
model_prepared = quantize_fx.prepare_qat_fx(model_fused, qconfig_mapping, torch.randn(1, 7, 256, 192))
# set training criterion
criterionL1 = nn.L1Loss()
criterionVGG = VGGLoss()
criterionL2 = nn.MSELoss('sum')

# set training optimizer
params_warp = [p for p in PF_warp_model.parameters()]
params_gen = [p for p in model_prepared.parameters()]
optimizer_warp = torch.optim.Adam(params_warp, lr=0.2 * opt.lr, betas=(opt.beta1, 0.999))
optimizer_gen = torch.optim.Adam(params_gen, lr=opt.lr, betas=(opt.beta1, 0.999))

total_steps = (start_epoch - 1) * dataset_size + epoch_iter

step = 0
step_per_batch = dataset_size

#--------------------------training Gen_Model-----------------------

# finetune warp_model and gen_model
# replace corresponding modules
PF_warp_model.train()

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):

    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size

    for i, data in enumerate(train_loader):

        iter_start_time = time.time()

        total_steps += 1
        epoch_iter += 1
        save_fake = True

        t_mask = torch.FloatTensor((data['label'].cpu().numpy() == 7).astype(np.float))
        data['label'] = data['label'] * (1 - t_mask) + t_mask * 4
        edge = data['edge']
        pre_clothes_edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int))
        clothes = data['color']
        clothes = clothes * pre_clothes_edge
        edge_un = data['edge_un']
        pre_clothes_edge_un = torch.FloatTensor((edge_un.detach().numpy() > 0.5).astype(np.int))
        clothes_un = data['color_un']
        clothes_un = clothes_un * pre_clothes_edge_un
        person_clothes_edge = torch.FloatTensor((data['label'].cpu().numpy() == 4).astype(np.int))
        real_image = data['image']
        person_clothes = real_image * person_clothes_edge
        pose = data['pose']
        size = data['label'].size()
        oneHot_size1 = (size[0], 25, size[2], size[3])
        densepose = torch.cuda.FloatTensor(torch.Size(oneHot_size1)).zero_()
        densepose = densepose.scatter_(1, data['densepose'].data.long().cuda(), 1.0)
        densepose_fore = data['densepose'] / 24
        face_mask = torch.FloatTensor((data['label'].cpu().numpy() == 1).astype(np.int)) + torch.FloatTensor((data['label'].cpu().numpy() == 12).astype(np.int))
        other_clothes_mask = torch.FloatTensor((data['label'].cpu().numpy() == 5).astype(np.int)) + torch.FloatTensor((data['label'].cpu().numpy() == 6).astype(np.int)) \
                            + torch.FloatTensor((data['label'].cpu().numpy() == 8).astype(np.int)) + torch.FloatTensor((data['label'].cpu().numpy() == 9).astype(np.int)) \
                            + torch.FloatTensor((data['label'].cpu().numpy() == 10).astype(np.int))
        face_img = face_mask * real_image
        other_clothes_img = other_clothes_mask * real_image
        preserve_mask = torch.cat([face_mask, other_clothes_mask], 1)

        concat_un = torch.cat([preserve_mask.cuda(), densepose, pose.cuda()], 1)
        flow_out_un = PB_warp_model(concat_un.cuda(), clothes_un.cuda(), pre_clothes_edge_un.cuda())
        warped_cloth_un, last_flow_un, cond_un_all, flow_un_all, delta_list_un, x_all_un, x_edge_all_un, delta_x_all_un, delta_y_all_un = flow_out_un
        warped_prod_edge_un = F.grid_sample(pre_clothes_edge_un.cuda(), last_flow_un.permute(0, 2, 3, 1),
                                            mode='bilinear', padding_mode='zeros')

        flow_out_sup = PB_warp_model(concat_un.cuda(), clothes.cuda(), pre_clothes_edge.cuda())
        warped_cloth_sup, last_flow_sup, cond_sup_all, flow_sup_all, delta_list_sup, x_all_sup, x_edge_all_sup, delta_x_all_sup, delta_y_all_sup = flow_out_sup

        arm_mask = torch.FloatTensor((data['label'].cpu().numpy() == 11).astype(np.float)) + torch.FloatTensor((data['label'].cpu().numpy() == 13).astype(np.float))
        hand_mask = torch.FloatTensor((data['densepose'].cpu().numpy() == 3).astype(np.int)) + torch.FloatTensor((data['densepose'].cpu().numpy() == 4).astype(np.int))
        dense_preserve_mask = torch.FloatTensor((data['densepose'].cpu().numpy() == 15).astype(np.int)) + torch.FloatTensor((data['densepose'].cpu().numpy() == 16).astype(np.int)) \
                            + torch.FloatTensor((data['densepose'].cpu().numpy() == 17).astype(np.int)) + torch.FloatTensor((data['densepose'].cpu().numpy() == 18).astype(np.int)) \
                            + torch.FloatTensor((data['densepose'].cpu().numpy() == 19).astype(np.int)) + torch.FloatTensor((data['densepose'].cpu().numpy() == 20).astype(np.int)) \
                            + torch.FloatTensor((data['densepose'].cpu().numpy() == 21).astype(np.int)) + torch.FloatTensor((data['densepose'].cpu().numpy() == 22))
        hand_img = (arm_mask * hand_mask) * real_image
        dense_preserve_mask = dense_preserve_mask.cuda() * (1 - warped_prod_edge_un)
        preserve_region = face_img + other_clothes_img + hand_img

        gen_inputs_un = torch.cat([preserve_region.cuda(), warped_cloth_un, warped_prod_edge_un, dense_preserve_mask], 1)
        gen_outputs_un = PB_gen_model(gen_inputs_un)
        p_rendered_un, m_composite_un = torch.split(gen_outputs_un, [3, 1], 1)
        p_rendered_un = torch.tanh(p_rendered_un)
        m_composite_un = torch.sigmoid(m_composite_un)
        m_composite_un = m_composite_un * warped_prod_edge_un
        p_tryon_un = warped_cloth_un * m_composite_un + p_rendered_un * (1 - m_composite_un)

        flow_out = PF_warp_model(p_tryon_un.detach(), clothes.cuda(), pre_clothes_edge.cuda())
        warped_cloth, last_flow, cond_all, flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all = flow_out
        warped_prod_edge = x_edge_all[4]

        epsilon = 0.001
        loss_smooth = sum([TVLoss(x) for x in delta_list])
        loss_warp = 0
        loss_fea_sup_all = 0
        loss_flow_sup_all = 0

        l1_loss_batch = torch.abs(warped_cloth_sup.detach() - person_clothes.cuda())
        l1_loss_batch = l1_loss_batch.reshape(opt.batchSize, 3 * 256 * 192)
        l1_loss_batch = l1_loss_batch.sum(dim=1) / (3 * 256 * 192)
        l1_loss_batch_pred = torch.abs(warped_cloth.detach() - person_clothes.cuda())
        l1_loss_batch_pred = l1_loss_batch_pred.reshape(opt.batchSize, 3 * 256 * 192)
        l1_loss_batch_pred = l1_loss_batch_pred.sum(dim=1) / (3 * 256 * 192)
        weight = (l1_loss_batch < l1_loss_batch_pred).float()
        num_all = len(np.where(weight.cpu().numpy() > 0)[0])
        if num_all == 0:
            num_all = 1

        for num in range(5):
            cur_person_clothes = F.interpolate(person_clothes, scale_factor=0.5 ** (4 - num), mode='bilinear')
            cur_person_clothes_edge = F.interpolate(person_clothes_edge, scale_factor=0.5 ** (4 - num), mode='bilinear')
            loss_l1 = criterionL1(x_all[num], cur_person_clothes.cuda())
            loss_vgg = criterionVGG(x_all[num], cur_person_clothes.cuda())
            loss_edge = criterionL1(x_edge_all[num], cur_person_clothes_edge.cuda())
            b, c, h, w = delta_x_all[num].shape
            loss_flow_x = (delta_x_all[num].pow(2) + epsilon * epsilon).pow(0.45)
            loss_flow_x = torch.sum(loss_flow_x) / (b * c * h * w)
            loss_flow_y = (delta_y_all[num].pow(2) + epsilon * epsilon).pow(0.45)
            loss_flow_y = torch.sum(loss_flow_y) / (b * c * h * w)
            loss_second_smooth = loss_flow_x + loss_flow_y
            b1, c1, h1, w1 = cond_all[num].shape
            weight_all = weight.reshape(-1, 1, 1, 1).repeat(1, 256, h1, w1)
            cond_sup_loss = ((cond_sup_all[num].detach() - cond_all[num]) ** 2 * weight_all).sum() / (256 * h1 * w1 * num_all)
            loss_fea_sup_all = loss_fea_sup_all + (5 - num) * 0.04 * cond_sup_loss
            loss_warp = loss_warp + (num + 1) * loss_l1 + (num + 1) * 0.2 * loss_vgg + (num + 1) * 2 * loss_edge + (num + 1) * 6 * loss_second_smooth + (5 - num) * 0.04 * cond_sup_loss
            if num >= 2:
                b1, c1, h1, w1 = flow_all[num].shape
                weight_all = weight.reshape(-1, 1, 1).repeat(1, h1, w1)
                flow_sup_loss = (torch.norm(flow_sup_all[num].detach() - flow_all[num], p=2, dim=1) * weight_all).sum() / (h1 * w1 * num_all)
                loss_flow_sup_all = loss_flow_sup_all + (num + 1) * 1 * flow_sup_loss
                loss_warp = loss_warp + (num + 1) * 1 * flow_sup_loss

        loss_warp = 0.01 * loss_smooth + loss_warp

        skin_mask = warped_prod_edge_un.detach() * (1 - person_clothes_edge.cuda())
        gen_inputs = torch.cat([p_tryon_un.detach(), warped_cloth, warped_prod_edge], 1)
        gen_outputs = model_prepared(gen_inputs)
        p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        m_composite1 = m_composite * warped_prod_edge
        m_composite = person_clothes_edge.cuda() * m_composite1
        p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

        loss_mask_l1 = torch.mean(torch.abs(1 - m_composite))
        loss_l1_skin = criterionL1(p_rendered * skin_mask, skin_mask * real_image.cuda())
        loss_vgg_skin = criterionVGG(p_rendered * skin_mask, skin_mask * real_image.cuda())
        loss_l1 = criterionL1(p_tryon, real_image.cuda())
        loss_vgg = criterionVGG(p_tryon, real_image.cuda())
        bg_loss_l1 = criterionL1(p_rendered, real_image.cuda())
        bg_loss_vgg = criterionVGG(p_rendered, real_image.cuda())

        if epoch < opt.niter:
            loss_gen = (loss_l1 * 5 + loss_l1_skin * 30 + loss_vgg + loss_vgg_skin * 2 + bg_loss_l1 * 5 + bg_loss_vgg + 1 * loss_mask_l1)
        else:
            loss_gen = (loss_l1 * 5 + loss_l1_skin * 60 + loss_vgg + loss_vgg_skin * 4 + bg_loss_l1 * 5 + bg_loss_vgg + 1 * loss_mask_l1)

        loss_all = 0.25 * loss_warp + loss_gen

        optimizer_warp.zero_grad()
        optimizer_gen.zero_grad()
        loss_all.backward()
        optimizer_warp.step()
        optimizer_gen.step()

        step += 1
        iter_end_time = time.time()
        iter_delta_time = iter_end_time - iter_start_time
        step_delta = (step_per_batch - step % step_per_batch) + step_per_batch * (opt.niter + opt.niter_decay - epoch)
        eta = iter_delta_time * step_delta
        eta = str(datetime.timedelta(seconds=int(eta)))
        time_stamp = datetime.datetime.now()
        now = time_stamp.strftime('%Y.%m.%d-%H:%M:%S')

        if step % 100 == 0:
            if opt.local_rank == 0:
                print('{}:{}:[step-{}]--[loss-{:.6f}]--[loss-{:.6f}]--[ETA-{}]'.format(now, epoch_iter, step, loss_gen, loss_warp, eta))

        if epoch_iter >= dataset_size:
            break

    if epoch > opt.niter:
        PF_warp_model.update_learning_rate_warp(optimizer_warp)
        PF_warp_model.update_learning_rate(optimizer_gen)

# End of epoch
model_quantized = quantize_fx.convert_fx(model_prepared)

# save the quantized model
torch.save(PF_warp_model.state_dict(), os.path.join(FilePath, 'NewPF_warp_model.pth'))
torch.save(model_quantized.state_dict(), os.path.join(FilePath, 'gen_QAT_Pytorch.pth'))

