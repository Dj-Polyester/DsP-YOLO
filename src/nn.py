import torch
from torch import nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv
    
class LCAM(nn.Module):
    """Lightweight Channel Attention Module (LCAM) for enhancing feature representation."""

    def __init__(self, c_in, c_bottleneck = None, multi=False, shared = True):
        super().__init__()
        self.shared = shared
        ngroups = 1 if shared else 2
        c_ = ngroups * c_in
        if c_bottleneck is None:
            c_bottleneck = c_ // 16 if multi else 1
        self.conv1 = nn.Conv2d(c_, c_bottleneck, 1, groups=ngroups)
        self.conv2 = nn.Conv2d(c_bottleneck, c_, 1, groups=ngroups)

    def conv_relu(self, x):
        x = self.conv1(x)  # First convolution
        x = F.relu(x, inplace=True)  # Apply ReLU activation
        x = self.conv2(x)  # Second convolution
        return x

    def forward(self, x: torch.Tensor):
        x_max = x.max(dim=(2, 3), keepdim=True)  # Global average pooling
        x_mean = x.mean(dim=(2, 3), keepdim=True)  # Global average pooling
        x1, x2 = None, None
        if self.shared:
            x1 = self.conv_relu(x_max)  # Apply convolution and ReLU to max-pooled features
            x2 = self.conv_relu(x_mean)  # Apply convolution and ReLU to mean
        else:
            x_cat = torch.cat([x_max, x_mean], dim=1)  # Concatenate along channel dimension
            x_cat = self.conv_relu(x_cat)  # Apply convolution and ReLU
            x1, x2 = x_cat.chunk(2, dim=1)  # Split into two parts
        x_add = x1 + x2  # Element-wise addition
        y = F.sigmoid(x_add)  # Apply sigmoid activation to get attention weights
        return x * y  # Scale input by attention weights
    
class LDSAM(nn.Module):
    """Lightweight Depthwise Separable Attention Module (LDSAM) for enhancing feature representation."""

    def __init__(self, c_in = None, c_out = None, multi=False):
        if c_out is None:
            c_out = c_in if multi else 1
        super().__init__()
        self.conv = nn.Conv2d(2, c_out, 1)

    def forward(self, x: torch.Tensor):
        """Apply depthwise separable attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying depthwise separable attention.
        """
        x_max = x.max(dim=1)  # Global average pooling
        x_mean = x.mean(dim=1)  # Global average pooling

        x_cat = torch.cat([x_max, x_mean], dim=1)  # Concatenate along channel dimension
        x_cat = self.conv(x_cat)  # Convolution
        y = F.sigmoid(x_cat)  # Apply sigmoid activation to get attention weights
        return x * y  # Scale input by attention weights
    
class LCBHAM(nn.Module):
    """Lightweight Channel and Spatial Attention Hybrid Module (LCBHAM) for enhancing feature representation."""

    def __init__(
            self, 
            c_in, 
            c_bottleneck = None, 
            c_out = None, 
            multi_lcam=False,
            multi_ldsam=False,
            shared = True,
        ):
        super().__init__()
        self.conv = Conv(c_in, c_in, 3, 2, 1, act=nn.Hardswish(inplace=True))  # Depthwise convolution
        self.lcam = LCAM(c_in, c_bottleneck, multi_lcam, shared)  # Lightweight Channel Attention Module
        self.ldsam = LDSAM(c_in, c_out, multi_ldsam)  # Lightweight Depthwise Separable Attention Module
    def forward(self, x: torch.Tensor):
        x = self.conv(x)  # Apply depthwise convolution
        x = self.lcam(x)  # Apply lightweight channel attention
        x = self.ldsam(x)  # Apply lightweight depthwise separable attention
        return x  # Return the output tensor