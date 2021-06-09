import torch.nn.functional as F
from torch import nn
import torch
from backbones.tdnn import TDNNLayer
from backbones.wav2vec import ConvFeatureExtractionModel
import numpy as np
from backbones.ecapa import *
from backbones.cbam import *

class Fp32GroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, inputs):
        output = F.group_norm(
            inputs.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(inputs)
    
def norm_block(is_layer_norm, dim, affine=True, is_instance_norm=False):
    if is_layer_norm:
        mod = nn.Sequential(
            TransposeLast(),
            Fp32LayerNorm(dim, elementwise_affine=affine),
            TransposeLast(),
        )
    else:
        if is_instance_norm:
            mod = Fp32GroupNorm(dim, dim, affine=False) # instance norm
        else:
            mod = Fp32GroupNorm(1, dim, affine=affine)  # layer norm

    return mod

class MultiScaleConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        dropout=0.0,
        non_affine_group_norm=False,
        activation=nn.ReLU(),):
        super().__init__()
        
        def block(n_in, n_out, k, stride, padding=0):
            return nn.Sequential(
                nn.Conv1d(n_in, n_out, k, stride=stride, bias=False, padding=padding),
                nn.Dropout(p=dropout),
                norm_block(is_layer_norm=False, dim=n_out, affine=not non_affine_group_norm,
                           is_instance_norm=True), activation)
        
        def filterblock(n_in, n_out, k, stride, padding=0):
            return nn.Sequential(
                nn.Conv1d(n_in, n_out, k, stride=stride, bias=False, padding=padding),
                nn.Dropout(p=dropout),
                norm_block(is_layer_norm=False, dim=n_out, affine=not non_affine_group_norm,
                           is_instance_norm=True))
        
        self.conv_front = nn.ModuleList()
        
#         # multi-3: s=12
#         self.conv_front.append(nn.Sequential(block(1, 90, 24, 12, 0), block(90, 192, 5, 1, 2)))
#         self.conv_front.append(nn.Sequential(block(1, 90, 12, 6, 0), block(90, 160, 5, 2, 0)))
#         self.conv_front.append(nn.Sequential(block(1, 90, 8, 4, 0), block(90, 160, 5, 3, 0)))  
        
#         # multi-3: s=18
#         self.conv_front.append(nn.Sequential(block(1, 90, 36, 18, 0), block(90, 192, 5, 1, 2)))
#         self.conv_front.append(nn.Sequential(block(1, 90, 18, 9, 0), block(90, 160, 5, 2, 0)))
#         self.conv_front.append(nn.Sequential(block(1, 90, 12, 6, 0), block(90, 160, 5, 3, 0)))  
        
#         # multi-3
#         self.conv_front.append(nn.Sequential(block(1, 90, 48, 24, 0), block(90, 192, 5, 1, 2)))
#         self.conv_front.append(nn.Sequential(block(1, 90, 16, 8, 0), block(90, 160, 5, 3, 0)))
#         self.conv_front.append(nn.Sequential(block(1, 90, 8, 4, 0), block(90, 160, 5, 6, 0)))  
        
        # multi-3
        self.conv_front.append(nn.Sequential(block(1, 50, 48, 24, 0), block(50, 192, 5, 1, 2)))
        self.conv_front.append(nn.Sequential(block(1, 50, 16, 8, 0), block(50, 160, 5, 3, 0)))
        self.conv_front.append(nn.Sequential(block(1, 50, 8, 4, 0), block(50, 160, 5, 6, 0)))  
        
#         # multi-5
#         self.conv_front.append(nn.Sequential(block(1, 54, 48, 24, 0), block(54, 112, 5, 1, 2)))
#         self.conv_front.append(nn.Sequential(block(1, 54, 24, 12, 0), block(54, 100, 5, 2, 1))) 
#         self.conv_front.append(nn.Sequential(block(1, 54, 16, 8, 0), block(54, 100, 5, 3, 0)))
#         self.conv_front.append(nn.Sequential(block(1, 54, 12, 6, 0), block(54, 100, 5, 4, 0))) 
#         self.conv_front.append(nn.Sequential(block(1, 54, 8, 4, 0), block(54, 100, 5, 6, 0)))  
        
        self.skip1 = nn.MaxPool1d(kernel_size=5, stride=8)
        self.skip2 = nn.MaxPool1d(kernel_size=3, stride=4, padding=1)
        
        self.conv1 = block(512, 512, 5, 2)
        self.conv2 = block(512, 512, 3, 2)
        self.conv3 = block(512, 512, 3, 2, padding=2)

    def forward(self, x):
        # BxT -> BxCxT
        
        # wave encoder
        enc = []
        ft_shape = []
        for conv in self.conv_front:
            enc.append(conv(x))
            ft_shape.append(conv(x).shape[-1])
            
        ft_max = np.min(np.array(ft_shape))
#         enc = torch.cat((enc[0][:, :, :ft_max], enc[1][:, :, :ft_max], 
#                          enc[2][:, :, :ft_max], enc[3][:, :, :ft_max],
#                          enc[4][:, :, :ft_max]), dim=1)
        enc = torch.cat((enc[0][:, :, :ft_max], enc[1][:, :, :ft_max], 
                         enc[2][:, :, :ft_max]), dim=1)
        
        # skipping layers
        skip1_out = self.skip1(enc)
        out = self.conv1(enc)
        skip2_out = self.skip2(out)
        out = self.conv2(out)
#         skip3_out = self.skip3(out)
        out = self.conv3(out)
        
        t_max = np.min(np.array([skip1_out.shape[-1], skip2_out.shape[-1], out.shape[-1]]))
        
        out = torch.cat((skip1_out[:, :, :t_max], skip2_out[:, :, :t_max], out[:, :, :t_max]), dim=1)

        return out
    
class TDNN_Block(nn.Module):
    def __init__(self, input_dim, output_dim=512, context_size=5, dilation=1, norm='bn', affine=True):
        super(TDNN_Block, self).__init__()
        if norm == 'bn':
            norm_layer = nn.BatchNorm1d(output_dim, affine=affine)
        elif norm == 'ln':
#             norm_layer = nn.GroupNorm(1, output_dim, affine=affine)
            norm_layer = Fp32GroupNorm(1, output_dim, affine=affine)
        elif norm == 'in':
            norm_layer = nn.GroupNorm(output_dim, output_dim, affine=False)
        else:
            raise ValueError('Norm should be {bn, ln, in}.')
        self.tdnn_layer = nn.Sequential(
            TDNNLayer(input_dim, output_dim, context_size, dilation),
            norm_layer,
            nn.ReLU()
        )
    def forward(self, x):
        return self.tdnn_layer(x)

class xvecTDNN(nn.Module):
    def __init__(self, feature_dim=512, embed_dim=512, norm='bn', p_dropout=0.0):
        super(xvecTDNN, self).__init__()
        self.tdnn = nn.Sequential(
            TDNN_Block(feature_dim, 512, 5, 1, norm=norm),
            TDNN_Block(512, 512, 3, 2, norm=norm),
            TDNN_Block(512, 512, 3, 3, norm=norm),
            TDNN_Block(512, 512, 1, 1, norm=norm),
            TDNN_Block(512, 1500, 1, 1, norm=norm),
        )

        self.fc1 = nn.Linear(3000, 512)
        self.bn = nn.BatchNorm1d(512)
        self.dropout_fc1 = nn.Dropout(p=p_dropout)
        self.lrelu = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(512, embed_dim)

    def forward(self, x):
        # Note: x must be (batch_size, feat_dim, chunk_len)
        x = self.tdnn(x)
        
        stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)
        
        x = self.dropout_fc1(self.lrelu(self.bn(self.fc1(stats))))
        x = self.fc2(x)
        
        return x

class xvecETDNN(nn.Module):
    def __init__(self, feature_dim=512, embed_dim=512, norm='bn', p_dropout=0.0):
        super(xvecETDNN, self).__init__()
        self.tdnn = nn.Sequential(
            TDNN_Block(feature_dim, 512, 5, 1, norm=norm),
            TDNN_Block(512, 512, 1, 1, norm=norm),
            TDNN_Block(512, 512, 3, 2, norm=norm),
            TDNN_Block(512, 512, 1, 1, norm=norm),
            TDNN_Block(512, 512, 3, 3, norm=norm),
            TDNN_Block(512, 512, 1, 1, norm=norm),
            TDNN_Block(512, 512, 3, 4, norm=norm),
            TDNN_Block(512, 512, 1, 1, norm=norm),
            TDNN_Block(512, 512, 1, 1, norm=norm),
            TDNN_Block(512, 1500, 1, 1, norm=norm),
        )

        self.fc1 = nn.Linear(3000, 512)
        self.bn = nn.BatchNorm1d(512)
        self.dropout_fc1 = nn.Dropout(p=p_dropout)
        self.lrelu = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(512, embed_dim)

    def forward(self, x):
        # Note: x must be (batch_size, feat_dim, chunk_len)
        x = self.tdnn(x)
        
        stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)
        
        x = self.dropout_fc1(self.lrelu(self.bn(self.fc1(stats))))
        x = self.fc2(x)
        
        return x
    
class ECAPATDNN(nn.Module):
    def __init__(self, in_channels=80, channels=512, embd_dim=192):
        super().__init__()
        self.layer1 = Conv1dReluBn(in_channels, channels, kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=2, dilation=2, scale=8)
        self.layer3 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=3, dilation=3, scale=8)
        self.layer4 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=4, dilation=4, scale=8)

        cat_channels = channels * 3
        self.conv = nn.Conv1d(cat_channels, cat_channels, kernel_size=1)
        self.pooling = AttentiveStatsPool(cat_channels, 128)
        self.bn1 = nn.BatchNorm1d(cat_channels * 2)
        self.linear = nn.Linear(cat_channels * 2, embd_dim)
        self.bn2 = nn.BatchNorm1d(embd_dim)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1) + out1
        out3 = self.layer3(out1 + out2) + out1 + out2
        out4 = self.layer4(out1 + out2 + out3) + out1 + out2 + out3

        out = torch.cat([out2, out3, out4], dim=1)
        out = F.relu(self.conv(out))
        out = self.bn1(self.pooling(out))
        out = self.bn2(self.linear(out))
        return out

class architecture(nn.Module):
    def __init__(self, embed_dim=512):
        super(architecture, self).__init__()
        
        self.feature_encoder = MultiScaleConvFeatureExtractionModel()
        self.tdnn_aggregator = xvecTDNN(feature_dim=512*3, embed_dim=128, norm='ln')

    def forward(self, x):
        out = self.feature_encoder(x)
        out = self.tdnn_aggregator(out)
        
        return out