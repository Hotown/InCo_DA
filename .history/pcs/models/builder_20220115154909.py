from typing import Optional
import torch
import torch.nn as nn
from .head import Classifier as ClassifierBase

class FixCL(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes:int, bottleneck_dim: Optional[int]=512, mlp=False, **kwargs):
        """
        dim: feature dimension (default: 512)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        if mlp:
            dim_mlp = backbone.out_features
            bottleneck = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp),
                nn.ReLU(),
                nn.Linear(dim_mlp, bottleneck_dim)
            )
        else:
            bottleneck = nn.Linear(backbone.out_features, bottleneck_dim)
            
        head_dim = 256
        
        head = nn.Sequential(
            nn.Linear(backbone.out_features, head_dim),
            nn.BatchNorm1d(head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, num_classes)
        )
        
        super(FixCL, self).__init__(backbone, num_classes, bottleneck=bottleneck, bottleneck_dim=bottleneck_dim, head=head,**kwargs)
        
        