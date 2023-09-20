import torch
import torch.nn as nn
from afwm import AFWM
from networks import ResUnetGenerator, VGGLoss
import torch.nn.functional as F
import numpy as np

class AFWNET(nn.Module):
    def __init__(self, opt):
        super(AFWNET, self).__init__()
        self.warp_model = AFWM(opt, 3)
        self.gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=torch.nn.BatchNorm2d)
        
    def forward(self, real_image, clothes, edge):
        edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int32))
        clothes = clothes * edge
        warped_cloth, last_flow = self.warp_model(real_image, clothes, edge)
        warped_edge = F.grid_sample(edge, last_flow.permute(0, 2, 3, 1),
                        mode='bilinear', padding_mode='zeros')

        gen_inputs = torch.cat([real_image, warped_cloth, warped_edge], 1)
        gen_outputs = self.gen_model(gen_inputs)
        p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
        p_rendered = torch.tanh(p_rendered)
        self.p_rendered = p_rendered
        m_composite = torch.sigmoid(m_composite)
        m_composite = m_composite * warped_edge
        p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)
        return p_tryon