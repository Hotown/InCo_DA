from typing import Optional
import torch
import torch.nn as nn
from .head import Classifier as ClassifierBase
from typing import List, Dict
from utils import torchutils

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
            
        torchutils.weights_init(bottleneck)
        
        head_dim = 256
        
        head = nn.Sequential(
            nn.Linear(backbone.out_features, head_dim),
            # nn.BatchNorm1d(head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, num_classes)
        )
        
        torchutils.weights_init(head)
        
        super(FixCL, self).__init__(backbone, num_classes, bottleneck=bottleneck, bottleneck_dim=bottleneck_dim, head=head,**kwargs)
        
class DomainDiscriminator(nn.Sequential):
    r"""Domain discriminator model from
    `Domain-Adversarial Training of Neural Networks (ICML 2015) <https://arxiv.org/abs/1505.07818>`_
    Distinguish whether the input features come from the source domain or the target domain.
    The source domain label is 1 and the target domain label is 0.
    Args:
        in_feature (int): dimension of the input feature
        hidden_size (int): dimension of the hidden features
        batch_norm (bool): whether use :class:`~torch.nn.BatchNorm1d`.
            Use :class:`~torch.nn.Dropout` if ``batch_norm`` is False. Default: True.
    Shape:
        - Inputs: (minibatch, `in_feature`)
        - Outputs: :math:`(minibatch, 1)`
    """

    def __init__(self, in_feature: int, hidden_size: int, batch_norm=True):
        if batch_norm:
            super(DomainDiscriminator, self).__init__(
                nn.Linear(in_feature, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )
        else:
            super(DomainDiscriminator, self).__init__(
                nn.Linear(in_feature, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )

    def get_parameters(self) -> List[Dict]:
        return [{"params": self.parameters(), "lr": 1.}]