import torch
import inspect
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Img2Col(nn.Module):
    def __init__(self, max_disp, dila_patch=1):
        """
        Arguments:
        - max_disp: maximum displacement
        - dila_patch: dilation on patch
        """
        super(Img2Col, self).__init__()

        patch_size = max_disp * 2 // dila_patch + 1
        pad_l = pad_t = pad_r = pad_b = max_disp

        self.patch_size = patch_size
        self.pad_size = (pad_l, pad_r, pad_t, pad_b)

        meshgrid_need_index = "indexing" in inspect.getfullargspec(torch.meshgrid).kwonlyargs
        meshgrid_kwargs = {"indexing": "ij"} if meshgrid_need_index else {}
        oy, ox = torch.meshgrid(
            torch.arange(0, patch_size) * dila_patch, 
            torch.arange(0, patch_size) * dila_patch, 
            **meshgrid_kwargs
        )
        oy = oy.flatten()
        ox = ox.flatten()

        self.register_buffer("oy", oy, persistent=False)
        self.register_buffer("ox", ox, persistent=False)
    
    @property
    def out_channels(self):
        return self.patch_size ** 2
    
    def im2col(self, img, H0, W0, stride=1):
        N, H, W, C = img.shape
        out_h = (H - H0) // stride + 1
        out_w = (W - W0) // stride + 1
        col = np.empty((N * out_h * out_w, H0 * W0 * C))
        outsize = out_w * out_h
        for y in range(out_h):
            y_min = y * stride
            y_max = y_min + H0
            y_start = y * out_w
            for x in range(out_w):
                x_min = x * stride
                x_max = x_min + W0
                col[y_start+x::outsize, :] = img[:, y_min:y_max, x_min:x_max, :].reshape(N, -1)
        return col
    
    def forward(self, fmap0: torch.Tensor, fmap1: torch.Tensor):
        # expand tensor fmap0
        fmap0 = fmap0.permute(0, 2, 3, 1).detach().numpy()
        N0, H0, W0, C0 = fmap0.shape
        fmap0 = fmap0.reshape(N0, -1)
        # padding fmap1 and expand tensor fmap1_pad
        fmap1_pad = F.pad(fmap1, self.pad_size, "constant", 0.0)
        fmap1_pad = fmap1_pad.permute(0, 2, 3, 1).detach().numpy()
        col = self.im2col(fmap1_pad, H0, W0)
        corr = [np.sum(np.multiply(col[i].reshape(1, -1), fmap0[i*N0 // col.shape[0]]).reshape(H0, W0, C0), axis=2) for i in range(col.shape[0])]
        corr = np.stack(corr, axis=0)

        return torch.from_numpy(corr.reshape(N0, -1, H0, W0)).float()