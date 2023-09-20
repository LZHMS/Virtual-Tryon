import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from corr_pure_torch import CorrTorch

def bilinear_grid_sample(im: Tensor,
                         grid: Tensor,
                         align_corners: bool = False) -> Tensor:
    n, c, h, w = im.shape
    gn, gh, gw, _ = grid.shape
    assert n == gn

    x = grid[:, :, :, 0]
    y = grid[:, :, :, 1]

    if align_corners:
        x = ((x + 1) / 2) * (w - 1)
        y = ((y + 1) / 2) * (h - 1)
    else:
        x = ((x + 1) * w - 1) / 2
        y = ((y + 1) * h - 1) / 2

    x = x.view(n, -1)
    y = y.view(n, -1)

    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = x0 + 1
    y1 = y0 + 1

    wa = ((x1 - x) * (y1 - y)).unsqueeze(1)
    wb = ((x1 - x) * (y - y0)).unsqueeze(1)
    wc = ((x - x0) * (y1 - y)).unsqueeze(1)
    wd = ((x - x0) * (y - y0)).unsqueeze(1)

    # Apply default for grid_sample function zero padding
    im_padded = F.pad(im, pad=[1, 1, 1, 1], mode='replicate')
    padded_h = h + 2
    padded_w = w + 2
    # save points positions after padding
    x0, x1, y0, y1 = x0 + 1, x1 + 1, y0 + 1, y1 + 1

    # Clip coordinates to padded image size
    x0 = torch.where(x0 < 0, torch.tensor(0), x0)
    x0 = torch.where(x0 > padded_w - 1, torch.tensor(padded_w - 1), x0)
    x1 = torch.where(x1 < 0, torch.tensor(0), x1)
    x1 = torch.where(x1 > padded_w - 1, torch.tensor(padded_w - 1), x1)
    y0 = torch.where(y0 < 0, torch.tensor(0), y0)
    y0 = torch.where(y0 > padded_h - 1, torch.tensor(padded_h - 1), y0)
    y1 = torch.where(y1 < 0, torch.tensor(0), y1)
    y1 = torch.where(y1 > padded_h - 1, torch.tensor(padded_h - 1), y1)

    im_padded = im_padded.view(n, c, -1)

    x0_y0 = (x0 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x0_y1 = (x0 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y0 = (x1 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y1 = (x1 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)

    Ia = torch.gather(im_padded, 2, x0_y0)
    Ib = torch.gather(im_padded, 2, x0_y1)
    Ic = torch.gather(im_padded, 2, x1_y0)
    Id = torch.gather(im_padded, 2, x1_y1)

    return (Ia * wa + Ib * wb + Ic * wc + Id * wd).reshape(n, c, gh, gw)

def apply_offset(offset):
    sizes = list(offset.size()[2:])
    grid_list = torch.meshgrid([torch.arange(size, device=offset.device) for size in sizes])
    grid_list = reversed(grid_list)

    grid_list = [grid.float().unsqueeze(0) + offset[:, dim, ...]
                 for dim, grid in enumerate(grid_list)]

    grid_list = [grid / ((size - 1.0) / 2.0) - 1.0
                 for grid, size in zip(grid_list, reversed(sizes))]

    return torch.stack(grid_list, dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return self.block(x) + x


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        )

    def forward(self, x):
        return self.block(x)


class FeatureEncoder(nn.Module):
    def __init__(self, in_channels, chns=[64, 128, 256, 256, 256]):
        super(FeatureEncoder, self).__init__()
        self.encoders = []
        for i, out_chns in enumerate(chns):
            if i == 0:
                encoder = nn.Sequential(DownSample(in_channels, out_chns),
                                        ResBlock(out_chns),
                                        ResBlock(out_chns))
            else:
                encoder = nn.Sequential(DownSample(chns[i - 1], out_chns),
                                        ResBlock(out_chns),
                                        ResBlock(out_chns))

            self.encoders.append(encoder)

        self.encoders = nn.ModuleList(self.encoders)

    def forward(self, x):
        encoder_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoder_features.append(x)
            
        return encoder_features


class RefinePyramid(nn.Module):
    def __init__(self, chns=[64, 128, 256, 256, 256], fpn_dim=256):
        super(RefinePyramid, self).__init__()
        self.chns = chns

        self.adaptive = []
        for in_chns in list(reversed(chns)):
            adaptive_layer = nn.Conv2d(in_chns, fpn_dim, kernel_size=1)
            self.adaptive.append(adaptive_layer)
        self.adaptive = nn.ModuleList(self.adaptive)

        self.smooth = []
        for i in range(len(chns)):
            smooth_layer = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1)
            self.smooth.append(smooth_layer)
        self.smooth = nn.ModuleList(self.smooth)

    def forward(self, x):
        x = list(x)
        feature_list = []
        last_feature = None

        for layer1, layer2 in zip(enumerate(self.adaptive), enumerate(self.smooth)):
            feature = layer1[1](x[-1-layer1[0]])

            if last_feature is not None:
                feature = feature + F.interpolate(last_feature, scale_factor=2.0, mode='nearest')
            feature = layer2[1](feature)
            last_feature = feature
            feature_list.append(feature)

        return list(reversed(feature_list))

    
class AFlowNet(nn.Module):
    def __init__(self, num_pyramid, fpn_dim=256):
        super(AFlowNet, self).__init__()
        self.netMain = []
        self.netRefine = []
        for i in range(num_pyramid):
            netMain_layer = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=49, out_channels=128, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)
            )

            netRefine_layer = torch.nn.Sequential(
                torch.nn.Conv2d(2 * fpn_dim, out_channels=128, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)
            )
            self.netMain.append(netMain_layer)
            self.netRefine.append(netRefine_layer)

        self.netMain = nn.ModuleList(self.netMain)
        self.netRefine = nn.ModuleList(self.netRefine)
        
    def forward(self, x, x_warps, x_conds, warp_feature=True):        # cloth, cloth_feature, pose_feature
        last_flow = None
        for i in range(len(x_warps)):
            x_warp = x_warps[-1-i]
            x_cond = x_conds[-1-i]

            if last_flow is not None and warp_feature:
                x_warp_after = bilinear_grid_sample(x_warp, last_flow.detach().permute(0, 2, 3, 1),
                                             align_corners=False)
            else:
                x_warp_after = x_warp

            max_displacement = 3
            stride2 = 1     # 跳步
            kernel = CorrTorch(max_disp=max_displacement, dila_patch=stride2)
            # kernel = torch.jit.script(kernel)

            corre_out = kernel(x_warp_after, x_cond)
            tenCorrelation = F.leaky_relu(
                input=corre_out,
                negative_slope=0.1, inplace=False)

            flow = self.netMain[i](tenCorrelation)
            flow = apply_offset(flow)

            if last_flow is not None:
                flow = bilinear_grid_sample(last_flow, flow, align_corners=False)
            else:
                flow = flow.permute(0, 3, 1, 2)

            last_flow = flow
            x_warp = bilinear_grid_sample(x_warp, flow.permute(0, 2, 3, 1), align_corners=False)
            concat = torch.cat([x_warp, x_cond], 1)
            flow = self.netRefine[i](concat)
            flow = apply_offset(flow)
            flow = bilinear_grid_sample(last_flow, flow, align_corners=False)

            last_flow = F.interpolate(flow, scale_factor=2, mode='bilinear')

        x_warp = bilinear_grid_sample(x, last_flow.permute(0, 2, 3, 1),
                               align_corners=False)
        return x_warp, last_flow

class AFWM(nn.Module):

    def __init__(self, opt, input_nc):
        super(AFWM, self).__init__()
        num_filters = [64, 128, 256, 256, 256]
        self.image_features = FeatureEncoder(3, num_filters)
        self.cond_features = FeatureEncoder(input_nc, num_filters)
        self.image_FPN = RefinePyramid(num_filters)
        self.cond_FPN = RefinePyramid(num_filters)
        self.aflow_net = AFlowNet(len(num_filters))
        #self.quant = torch.quantization.QuantStub()
        #self.dequant = torch.quantization.DeQuantStub()

    def forward(self, cond_input, image_input):    # real_img, cloth
        #cond_input, image_input = self.quant(cond_input), self.quant(image_input)
        image_pyramids = self.image_FPN(self.image_features(image_input))
        cond_pyramids = self.cond_FPN(self.cond_features(cond_input))  # maybe use nn.Sequential

        x_warp, last_flow = self.aflow_net(image_input, image_pyramids, cond_pyramids)
        #return self.dequant(x_warp), self.dequant(last_flow)
        return x_warp, last_flow

# Following classes will be used for modole pruning
class IFPN(nn.Module):
    def __init__(self):
        super(IFPN, self).__init__()
        num_filters = [64, 128, 256, 256, 256]
        self.image_features = FeatureEncoder(3, num_filters)
        self.image_FPN = RefinePyramid(num_filters)

    def forward(self, image_input):
        return self.image_FPN(self.image_features(image_input))

class CFPN(nn.Module):
    def __init__(self, input_nc):
        super(CFPN, self).__init__()
        num_filters = [64, 128, 256, 256, 256]
        self.cond_features = FeatureEncoder(input_nc, num_filters)
        self.cond_FPN = RefinePyramid(num_filters)

    def forward(self, cond_input):
        return self.cond_FPN(self.cond_features(cond_input))

