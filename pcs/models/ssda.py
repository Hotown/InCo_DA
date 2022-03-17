import random

import numpy as np
import torch
import torch.cuda.comm
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils import reverse_domain, torchutils
from typing import Optional, Any, Tuple
from torch.autograd import Function

class Entropy(nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()
    
    def forward(self, input: torch.Tensor, mean=True) -> torch.Tensor:
        p = F.softmax(input, dim=1)
        en = -torch.sum(p * torch.log(p + 1e-5), dim=1)
        if mean:
            return torch.mean(en)
        else:
            return en
        
class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, size_average=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.size_average = size_average
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        if self.size_average:
            loss = (- targets * log_probs).mean(0).sum()
        else:
            loss = (- targets * log_probs).sum(1)
        return loss


class CrossEntropyMix(nn.Module):
    def __init__(self, num_classes, use_gpu=True, size_average=True):
        super(CrossEntropyMix, self).__init__()
        self.num_classes = num_classes
        self.use_gpu = use_gpu
        self.size_average = size_average
        self.logsoftmax = nn.LogSoftmax(dim=1)
    
    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        if self.use_gpu:
            targets = targets.cuda()
        if self.size_average:
            loss = (-targets * log_probs).mean(0).sum()
        else:
            loss = (- targets * log_probs).sum(1)
        return loss

class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class WarmStartGradientReverseLayer(nn.Module):
    """Gradient Reverse Layer :math:`\mathcal{R}(x)` with warm start
        The forward and backward behaviours are:
        .. math::
            \mathcal{R}(x) = x,
            \dfrac{ d\mathcal{R}} {dx} = - \lambda I.
        :math:`\lambda` is initiated at :math:`lo` and is gradually changed to :math:`hi` using the following schedule:
        .. math::
            \lambda = \dfrac{2(hi-lo)}{1+\exp(- α \dfrac{i}{N})} - (hi-lo) + lo
        where :math:`i` is the iteration step.
        Args:
            alpha (float, optional): :math:`α`. Default: 1.0
            lo (float, optional): Initial value of :math:`\lambda`. Default: 0.0
            hi (float, optional): Final value of :math:`\lambda`. Default: 1.0
            max_iters (int, optional): :math:`N`. Default: 1000
            auto_step (bool, optional): If True, increase :math:`i` each time `forward` is called.
              Otherwise use function `step` to increase :math:`i`. Default: False
        """

    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 1000., auto_step: Optional[bool] = False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """"""
        coeff = np.float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1

class DomainAdversarialLoss(nn.Module):
    r"""
    The Domain Adversarial Loss proposed in
    `Domain-Adversarial Training of Neural Networks (ICML 2015) <https://arxiv.org/abs/1505.07818>`_
    Domain adversarial loss measures the domain discrepancy through training a domain discriminator.
    Given domain discriminator :math:`D`, feature representation :math:`f`, the definition of DANN loss is
    .. math::
        loss(\mathcal{D}_s, \mathcal{D}_t) = \mathbb{E}_{x_i^s \sim \mathcal{D}_s} \text{log}[D(f_i^s)]
            + \mathbb{E}_{x_j^t \sim \mathcal{D}_t} \text{log}[1-D(f_j^t)].
    Args:
        domain_discriminator (torch.nn.Module): A domain discriminator object, which predicts the domains of features. Its input shape is (N, F) and output shape is (N, 1)
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
        grl (WarmStartGradientReverseLayer, optional): Default: None.
    Inputs:
        - f_s (tensor): feature representations on source domain, :math:`f^s`
        - f_t (tensor): feature representations on target domain, :math:`f^t`
        - w_s (tensor, optional): a rescaling weight given to each instance from source domain.
        - w_t (tensor, optional): a rescaling weight given to each instance from target domain.
    Shape:
        - f_s, f_t: :math:`(N, F)` where F means the dimension of input features.
        - Outputs: scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(N, )`.
    Examples::
        >>> from dalib.modules.domain_discriminator import DomainDiscriminator
        >>> discriminator = DomainDiscriminator(in_feature=1024, hidden_size=1024)
        >>> loss = DomainAdversarialLoss(discriminator, reduction='mean')
        >>> # features from source domain and target domain
        >>> f_s, f_t = torch.randn(20, 1024), torch.randn(20, 1024)
        >>> # If you want to assign different weights to each instance, you should pass in w_s and w_t
        >>> w_s, w_t = torch.randn(20), torch.randn(20)
        >>> output = loss(f_s, f_t, w_s, w_t)
    """

    def __init__(self, domain_discriminator: nn.Module, reduction='mean',
                 grl=None):
        super(DomainAdversarialLoss, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True) if grl is None else grl
        self.domain_discriminator = domain_discriminator
        self.bce = lambda input, target, weight: \
            F.binary_cross_entropy(input, target, weight=weight, reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor,
                w_s: Optional[torch.Tensor] = None, w_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        f = self.grl(torch.cat((f_s, f_t), dim=0))
        d = self.domain_discriminator(f)
        d_s, d_t = d.chunk(2, dim=0)
        d_label_s = torch.ones((f_s.size(0), 1)).to(f_s.device)
        d_label_t = torch.zeros((f_t.size(0), 1)).to(f_t.device)
        # self.domain_discriminator_accuracy = 0.5 * (binary_accuracy(d_s, d_label_s) + binary_accuracy(d_t, d_label_t))

        if w_s is None:
            w_s = torch.ones_like(d_label_s)
        if w_t is None:
            w_t = torch.ones_like(d_label_t)
        return 0.5 * (self.bce(d_s, d_label_s, w_s.view_as(d_s)) + self.bce(d_t, d_label_t, w_t.view_as(d_t)))

class SSDALossModule(torch.nn.Module):
    def __init__(self, config, gpu_devices):
        super(SSDALossModule, self).__init__()
        self.config = config
        self.gpu_devices = gpu_devices

        self.k = config.loss_params.k
        self.t = config.loss_params.temp
        self.m = config.loss_params.m
        self.loss_w = self.config.loss_params.weight
        self.loss = self.config.loss_params.loss

        self.indices = None
        self.outputs = None
        self.indices_random = None
        self.batch_size = None

        self.broadcast_name = []

    def get_attr(self, domain, name):
        return getattr(self, name + "_" + domain)

    def set_attr(self, domain, name, value):
        setattr(self, name + "_" + domain, value)
        return self.get_attr(domain, name)

    def set_broadcast(self, domain_name, name, value, comm=False):
        if isinstance(value, list):
            value = value.copy()
            for i in range(len(value)):
                value[i] = torch.cuda.comm.broadcast(value[i], self.gpu_devices)
        else:
            value = torch.cuda.comm.broadcast(value, self.gpu_devices)

        self.set_attr(domain_name, f"{name}_broadcast", value)
        self.set_attr(domain_name, name, None)
        self.broadcast_name.append((domain_name, name))

    def get_broadcast(self, domain_name, name, gpu_idx):
        broadcast = self.get_attr(domain_name, f"{name}_broadcast")
        if len(broadcast) == 0 or isinstance(broadcast[0], list):
            tmp = []
            for x in broadcast:
                tmp.append(x[gpu_idx])
            self.set_attr(domain_name, name, tmp)
        else:
            tmp = broadcast[gpu_idx]
            self.set_attr(domain_name, name, tmp)
        return tmp

    def _identify_gpu(self, gpu_idx):
        for domain_name, name in self.broadcast_name:
            self.get_broadcast(domain_name, name, gpu_idx)

    # memory bank calculation function
    @torch.no_grad()
    def updated_new_data_memory(self, domain, indices, outputs):
        """Compute new memory bank in indices by momentum

        Args:
            indices: indices of memory bank features to update
            outputs: output of features
            domain (str): 'source', 'target'
        """
        memory_bank = self.get_attr(domain, "memory_bank")
        data_memory = torch.index_select(memory_bank, 0, indices)

        outputs = F.normalize(outputs, dim=1)
        m = self.m
        new_data_memory = data_memory * m + (1 - m) * outputs
        return F.normalize(new_data_memory, dim=1)

    # @torch.no_grad()
    # def updated_new_proto_memory(self, domain, indices, outputs):
    #     """Compute new mix prototype memory by momentum

    #     """
    #     memory_bank_mix = self.get_attr(domain, "memory_bank_mix")
    #     data_memory = torch.index_select(memory_bank_mix, 0, indices)
    #     outputs = F.normalize(outputs, dim=1)
    #     m = self.m
    #     new_data_memory = data_memory
    def _get_Z(self, domain, vec, t):
        """Get denominator in ID

        Args:
            vec: output features [batch_size, dim]
            domain (str): 'source', 'target'

        Returns:
            [batch_size] denominator in ID
        """
        bank = self.get_attr(domain, "memory_bank")  # [data_len]
        Z = torchutils.contrastive_sim_z(vec, bank, tao=t)
        return Z

    def _get_all_dot_products(self, domain, vec):
        """get dot product with all vectors in memory bank

        Args:
            vec: [bs, dim]
            domain (str): 'source', 'target'

        Returns:
            [bs, data_len]
        """
        assert len(vec.size()) == 2
        bank = self.get_attr(domain, "memory_bank")
        return torch.matmul(vec, torch.transpose(bank, 1, 0))

    def _compute_I2M_loss(self, domain, loss_type, t=0.05):
        """Loss Instance with Mix domain
        """
        assert loss_type in ["cross", "tgt", "src"]
        loss = torch.Tensor([0]).cuda()
        if (loss_type == "tgt" and domain == "source") or (
            loss_type == "src" and domain == "target"
        ):
            return loss
        memory_bank_mix  = self.get_attr(reverse_domain(domain), "memory_bank_mix")
        outputs = self.outputs

        p = torchutils.contrastive_sim(outputs, memory_bank_mix, tao=t)
        z = torch.sum(p, dim=-1)
        p = p / z.unsqueeze(1)
        mix_loss = -torch.sum(p * torch.log(p)) / self.batch_size
        loss += mix_loss

        return loss
    
    def _compute_I2C_loss(self, domain, loss_type, t=0.05):
        """Loss CrossSelf in essay (Cross-domain Instance-Prototype SSL)

        Args:
            domain (str): 'source', 'target'
            loss_type (str, optional): 'each', 'all', 'src', 'tgt'. Defaults to "zero".
        """
        assert loss_type in ["cross", "tgt", "src"]
        loss = torch.Tensor([0]).cuda()
        if (loss_type == "tgt" and domain == "source") or (
            loss_type == "src" and domain == "target"
        ):
            return loss

        clus = "each"

        k_list = self.config.k_list
        n_kmeans = len(k_list)

        cluster_centroids = self.get_attr(
            reverse_domain(domain), f"cluster_centroids_{clus}"
        )
        outputs = self.outputs

        for each_k_idx, k in enumerate(k_list):
            centroids = cluster_centroids[each_k_idx]
            phi = t

            p = torchutils.contrastive_sim(outputs, centroids, tao=phi)
            z = torch.sum(p, dim=-1)  # [bs]
            p = p / z.unsqueeze(1)  # [bs, k]

            cur_loss = -torch.sum(p * torch.log(p)) / self.batch_size

            loss = loss + cur_loss

        loss /= n_kmeans

        return loss
        
    # TODO: use proto
    def _compute_proto_loss(self, domain, loss_type, t=0.05):
        loss = torch.Tensor([0]).cuda()
        if (loss_type.startswith("src") and domain == "target") or (
            loss_type.startswith("tgt") and domain == "source"
        ):
            return loss

        is_fix = "fix" in loss_type
        proto = self.get_attr(domain, "memory_bank_proto")
        labels = self.get_attr(domain, "labels")
        if domain == "target":
            labels = labels.squeeze_(0)
        batch_labels = labels[self.indices]
        batch_proto = proto[batch_labels]
        outputs = self.outputs
        # calculate similarity
        dot_exp = torch.exp(
            torch.sum(outputs * batch_proto, dim=-1) / t
        )
        assert not torch.isnan(outputs).any()
        assert not torch.isnan(proto).any()
        assert not torch.isnan(dot_exp).any()
        # calculate Z
        # all_phi = t if is_fix else phis.unsqueeze(0).repeat(outputs.shape[0], 1)
        z = torchutils.contrastive_sim_z(outputs, batch_proto, tao=t)

        # calculate loss
        p = dot_exp / z

        loss = loss - torch.sum(torch.log(p)) / p.size(0)

        return loss

    # def _compute_proto_loss(self, domain, loss_type, t=0.05):
    #     """Loss PC in essay (part of In-domain Prototypical Contrastive Learning)

    #     Args:
    #         domain (str): 'source', 'target'
    #         loss_type (str, optional): 'each', 'all', 'src', 'tgt'. Defaults to "zero".
    #     """
    #     loss = torch.Tensor([0]).cuda()
    #     if (loss_type.startswith("src") and domain == "target") or (
    #         loss_type.startswith("tgt") and domain == "source"
    #     ):
    #         return loss

    #     is_fix = "fix" in loss_type
    #     clus = "each"
    #     n_kmeans = self.config.loss_params.clus.n_kmeans
    #     k_list = self.config.k_list
    #     c_domain = domain

    #     cluster_labels = self.get_attr(domain, f"cluster_labels_{clus}")
    #     cluster_centroids = self.get_attr(c_domain, f"cluster_centroids_{clus}")
    #     cluster_phi = self.get_attr(c_domain, f"cluster_phi_{clus}")

    #     for each_k_idx, k in enumerate(k_list):
    #         # clus info
    #         labels = cluster_labels[each_k_idx]
    #         centroids = cluster_centroids[each_k_idx]
    #         phis = cluster_phi[each_k_idx]

    #         # batch info
    #         batch_labels = labels[self.indices]
    #         outputs = self.outputs
    #         batch_centroids = centroids[batch_labels]
    #         if loss_type == "fix":
    #             batch_phis = t
    #         else:
    #             batch_phis = phis[batch_labels]

    #         # calculate similarity
    #         dot_exp = torch.exp(
    #             torch.sum(outputs * batch_centroids, dim=-1) / batch_phis
    #         )

    #         assert not torch.isnan(outputs).any()
    #         assert not torch.isnan(batch_centroids).any()
    #         assert not torch.isnan(dot_exp).any()

    #         # calculate Z
    #         all_phi = t if is_fix else phis.unsqueeze(0).repeat(outputs.shape[0], 1)
    #         z = torchutils.contrastive_sim_z(outputs, centroids, tao=all_phi)

    #         # calculate loss
    #         p = dot_exp / z

    #         loss = loss - torch.sum(torch.log(p)) / p.size(0)

    #     loss /= n_kmeans

    #     return loss

    def _compute_CD_loss(self, domain, loss_type, t=0.05):
        """Loss CDS in essay arXiv:2003.08264v1, not used in essay

        Args:
            domain (str): different domain from current one
            loss_type (str): 'cross'

        Returns:
            CD loss
        """
        assert loss_type in ["cross"]

        bank = self.get_attr(reverse_domain(domain), "memory_bank")
        if self.config.loss_params.sample_ratio:
            num_sample = int(self.config.loss_params.sample_ratio * len(bank))
            sampled_index = random.sample(list(range(len(bank))), num_sample)
            sampled_index = torch.tensor(sampled_index)
            bank = bank[sampled_index]

        # [bs, data_len], numerator of P_{i',i}^{s->t}
        prods = torchutils.contrastive_sim(self.outputs, bank, tao=t)
        # [bs]
        z = torch.sum(prods, dim=-1)
        # [bs, data_len] P_{i',i}^{s->t}
        p = prods / z.unsqueeze(1)
        aux = p.max(dim=1)[1]
        # double sum
        loss = -torch.sum(p * torch.log(p)) / self.batch_size
        return loss[None,], aux
    
    def _compute_ID_loss(self, domain, loss_type, t=0.05):
        """Loss ID (Instance Discrimination), not used in essay.

        Args:
            domain (str): 'source', 'target'
            loss_type (str): 'each', 'all'

        Returns:
            ID loss
        """
        assert loss_type in ["all", "each", "src", "tgt"]

        loss = torch.Tensor([0]).cuda()
        if loss_type == "src":
            if domain == "target":
                return loss
            else:
                clus = "each"
        if loss_type == "tgt":
            if domain == "source":
                return loss
            else:
                clus = "each"

        bank = self.get_attr(domain, "memory_bank")

        memory_vecs = torch.index_select(bank, 0, self.indices)
        prods = torch.sum(memory_vecs * self.outputs, dim=-1)
        # [bs], numerator of P_i^s
        prods_exp = torch.exp(prods / t)
        # [bs], denominator of P_i^s

        if self.config.loss_params.sample_ratio:
            num_sample = int(self.config.loss_params.sample_ratio * len(bank))
            sampled_index = random.sample(list(range(len(bank))), num_sample)
            sampled_index = torch.tensor(sampled_index)
            sampled_bank = bank[sampled_index]
            ratio_inv = 1 / self.config.loss_params.sample_ratio
        else:
            sampled_bank = bank
            ratio_inv = 1

        Z = torchutils.contrastive_sim_z(self.outputs, sampled_bank, tao=t)
        Z = ratio_inv * Z
        if loss_type == "all":
            bank_rev = self.get_attr(reverse_domain(domain), "memory_bank")
            Z = Z + torchutils.contrastive_sim_z(self.outputs, bank_rev, tao=t)
        # [bs], P_i^s
        p = prods_exp / Z
        loss = -torch.sum(torch.log(p)) / self.batch_size
        return loss[
            None,
        ]

    def _compute_loss(self, domain, loss_type=None, t=0.05):
        loss_name, loss_args = loss_type.split("-")
        loss_fn = getattr(self, f"_compute_{loss_name}_loss")
        aux = None
        if loss_name in ["CD"]:
            loss, aux = loss_fn(domain, loss_type=loss_args, t=t)
        else:
            loss = loss_fn(domain, loss_type=loss_args, t=t)

        assert not torch.isinf(loss).any()
        if loss < 0:
            print(loss)
            print(loss_name)
            assert loss >= 0

        return loss, aux

    def forward(self, indices, outputs, domain, gpu_idx):
        self.indices = indices.detach()
        self.batch_size = self.indices.size(0)
        self.outputs = outputs
        self._identify_gpu(gpu_idx)

        loss_part = []
        loss = torch.zeros(1).cuda()
        aux_list = {}
        for i, ls in enumerate(self.loss):
            if (
                self.epoch <= self.config.loss_params.start[i]
                or self.epoch >= self.config.loss_params.end[i]
                or ls.split("-")[0] in ["cls", "tgt", "semi", "norm"]
            ):
                l = torch.zeros(1).cuda()
            else:
                l, aux = self._compute_loss(domain, ls, self.t[i])
                aux_list[ls] = aux
            loss_part.append(l)
            loss = loss + l

        new_data_memory = self.updated_new_data_memory(domain, indices, outputs)

        return loss, new_data_memory, loss_part, aux_list


def loss_info(feat, mb_feat, label, t=0.1):
    loss = torch.Tensor([0]).cuda()
    z = torchutils.contrastive_sim_z(feat, mb_feat, tao=t)
    for i, lb in enumerate(label):
        pos = mb_feat[label == lb]
        up = torch.exp(torchutils.dot(feat[i], pos) / t)
        p = up / z[i]
        assert all(p < 1)
        loss += -torch.sum(torch.log(p)) / len(p)
    loss /= len(feat)
    return loss


@torch.no_grad()
def update_data_memory(data_memory, outputs, m=0.9):
    outputs = F.normalize(outputs, dim=1)
    new_data_memory = data_memory * m + (1 - m) * outputs
    return F.normalize(new_data_memory, dim=1)
