import torch
import torch.nn as nn
from torchvision import models
import sys
import os

class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nn.ReLU(True)
        if norm_layer != None:
            downnorm = norm_layer(inner_nc)
            upnorm = norm_layer(outer_nc)

        if outermost:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            upconv = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downconv]
            up = [uprelu, upsample, upconv]
            model = down + [submodule] + up
        elif innermost:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            upconv = nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            if norm_layer == None:
                up = [uprelu, upsample, upconv]
            else:
                up = [uprelu, upsample, upconv, upnorm]
            model = down + up
        else:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            upconv = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            if norm_layer == None:
                down = [downrelu, downconv]
                up = [uprelu, upsample, upconv]
            else:
                down = [downrelu, downconv, downnorm]
                up = [uprelu, upsample, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class ResidualBlock(nn.Module):
    def __init__(self, in_features=48, norm_layer=nn.BatchNorm2d):
        super(ResidualBlock, self).__init__()
        self.relu = nn.ReLU(True)
        self.add_relu = torch.nn.quantized.FloatFunctional()
        if norm_layer == None:
            self.block = nn.Sequential(
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
                norm_layer(in_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
                norm_layer(in_features)
            )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out = self.add_relu.add_relu(out, residual)
        return out


class ResUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=48, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(ResUnetGenerator, self).__init__()
        unet_block = ResUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                                innermost=True)

        for i in range(num_downs - 5):
            unet_block = ResUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                    norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = ResUnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                                norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                                norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                                norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                                norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


class ResUnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(ResUnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=3,
                             stride=2, padding=1, bias=use_bias)

        res_downconv = [ResidualBlock(inner_nc, norm_layer), ResidualBlock(inner_nc, norm_layer)]
        res_upconv = [ResidualBlock(outer_nc, norm_layer), ResidualBlock(outer_nc, norm_layer)]

        downrelu = nn.ReLU(True)
        uprelu = nn.ReLU(True)
        if norm_layer != None:
            downnorm = norm_layer(inner_nc)
            upnorm = norm_layer(outer_nc)

        if outermost:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downconv, downrelu] + res_downconv
            up = [upsample, upconv]
            model = down + [submodule] + up
        elif innermost:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downconv, downrelu] + res_downconv
            if norm_layer == None:
                up = [upsample, upconv, uprelu] + res_upconv
            else:
                up = [upsample, upconv, upnorm, uprelu] + res_upconv
            model = down + up
        else:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            if norm_layer == None:
                down = [downconv, downrelu] + res_downconv
                up = [upsample, upconv, uprelu] + res_upconv
            else:
                down = [downconv, downnorm, downrelu] + res_downconv
                up = [upsample, upconv, upnorm, uprelu] + res_upconv

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


def load_checkpoint(model, checkpoint_path):
    checkpoint_path = os.path.join("checkpoints", checkpoint_path)
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")

    checkpoint_path = os.path.join(base_path, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    checkpoint_new = model.state_dict()
    for param in checkpoint_new:
        checkpoint_new[param] = checkpoint[param]

    model.load_state_dict(checkpoint_new)
    
class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 27):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
    
class VGGLoss(nn.Module):
    def __init__(self, layids = None):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.vgg.cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.layids = layids

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

def load_checkpoint_parallel(model, checkpoint_path):

    if not os.path.exists(checkpoint_path):
        print('No checkpoint!')
        return

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    checkpoint_new = model.state_dict()
    for param in checkpoint_new:
        checkpoint_new[param] = checkpoint[param]
    model.load_state_dict(checkpoint_new)


