import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.torchutils import grad_reverse
from torch.autograd import Function
from torchvision import models
from typing import Tuple, Optional, List, Dict

# __all__ = ['Classifier', 'CosineClassifier']

class Classifier(nn.Module):
    """A generic Classifier class for domain adaptation.
    Args:
        backbone (torch.nn.Module): Any backbone to extract 2-d features from data
        num_classes (int): Number of classes
        bottleneck (torch.nn.Module, optional): Any bottleneck layer. Use no bottleneck by default, *** we use it as project head. ***
        bottleneck_dim (int, optional): Feature dimension of the bottleneck layer. Default: -1
        head (torch.nn.Module, optional): Any classifier head. Use :class:`torch.nn.Linear` by default
        finetune (bool): Whether finetune the classifier or train from scratch. Default: True
    .. note::
        Different classifiers are used in different domain adaptation algorithms to achieve better accuracy
        respectively, and we provide a suggested `Classifier` for different algorithms.
        Remember they are not the core of algorithms. You can implement your own `Classifier` and combine it with
        the domain adaptation algorithm in this algorithm library.
    .. note::
        The learning rate of this classifier is set 10 times to that of the feature extractor for better accuracy
        by default. If you have other optimization strategies, please over-ride :meth:`~Classifier.get_parameters`.
    Inputs:
        - x (tensor): input data fed to `backbone`
    Outputs:
        - predictions: classifier's predictions
        - features: features after `bottleneck` layer and before `head` layer
    Shape:
        - Inputs: (minibatch, *) where * means, any number of additional dimensions
        - predictions: (minibatch, `num_classes`)
        - features: (minibatch, `features_dim`)
    """
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck: Optional[nn.Module]=None,
                 bottleneck_dim: Optional[int]=-1, head: Optional[nn.Module]=None, finetune=True, pool_layer=None):
        super(Classifier, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        # pool layer
        if pool_layer is None:
            self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
        else:
            self.pool_layer = pool_layer
        # bottleneck
        if bottleneck is None:
            self.bottleneck = nn.Identity()
            self._features_dim = backbone.out_features
        else:
            self.bottleneck = bottleneck
            assert bottleneck_dim > 0
            self._features_dim = bottleneck_dim
        # cls head
        if head is None:
            self.head = nn.Linear(backbone.out_features, num_classes)
        else:
            self.head = head
            
        self.finetune = finetune
    
    @property
    def features_dim(self) -> int:
        """The dimension of features in project head

        Returns:
            int: [dimension of features]
        """
        return self._features_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.pool_layer(self.backbone(x))
        feature = self.bottleneck(f)
        predictions = self.head(f)
        # if self.training:
        #     return feature, predictions
        # else:
        #     return predictions
        return f, feature, predictions
    
    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
        such as the relative learning rate of each layer

        Args:
            base_lr (float, optional): [learning rate]. Defaults to 1.0.

        Returns:
            List[Dict]: [description]
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
            {"params": self.head.parameters(), "lr": 1.0 * base_lr}
        ]
        
        return params


class CosineClassifier(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(CosineClassifier, self).__init__()
        self.fc = nn.Linear(inc, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1):
        self.normalize_fc()

        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc(x)
        x_out = x_out / self.temp

        return x_out

    def normalize_fc(self):
        self.fc.weight.data = F.normalize(self.fc.weight.data, p=2, eps=1e-12, dim=1)

    @torch.no_grad()
    def compute_discrepancy(self):
        self.normalize_fc()
        W = self.fc.weight.data
        D = torch.mm(W, W.transpose(0, 1))
        D_mask = 1 - torch.eye(self.num_class).cuda()
        return torch.sum(D * D_mask).item()
