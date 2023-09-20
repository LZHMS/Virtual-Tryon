import torch
import inspect
import torch.nn as nn
import torch.nn.functional as F

class CorrTorch(nn.Module):
    def __init__(self, max_disp, dila_patch=1):
        """
        Arguments:
        - max_disp: maximum displacement
        - dila_patch: dilation on patch
        """
        super(CorrTorch, self).__init__()

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

    def forward(self, fmap0: torch.Tensor, fmap1: torch.Tensor):
        fmap1_pad = F.pad(fmap1, self.pad_size, "constant", 0.0)

        _, _, H, W = fmap0.size()
        corr = [torch.sum(fmap0 * fmap1_pad[:, :, oyi:oyi+H, oxi:oxi+W], 
                          dim=1, keepdim=True) for oxi, oyi in zip(self.ox, self.oy)]
        corr = torch.cat(corr, dim=1)
        return corr
