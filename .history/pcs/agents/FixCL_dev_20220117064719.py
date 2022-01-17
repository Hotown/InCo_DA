from curses.ascii import BS
import os
import time
from tkinter import image_names
from xml.sax.handler import feature_external_ges

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from models import ( MemoryBank, torch_kmeans, update_data_memory, CrossEntropyLabelSmooth, Entropy)
from sklearn import metrics

from tqdm import tqdm
from utils import (AverageMeter, ProgressMeter, datautils, 
                    torchutils, get_model)
import models.builder

from . import BaseAgent

ls_abbr = {
    "cls-so": "cls",
    "proto-each": "P",
    "cls-info": "info",
    "I2C-cross": "C",
    "I2M-cross": "CM",
    "tgt-condentmax": "tCE",
    "tgt-entmin": "tE",
    "ID-each": "I",
    "CD-cross": "CD",
}


class DEVAgent(BaseAgent):
    def __init__(self, config):
        self.config = config
        self._define_task(config)
        self.is_features_computed = False
        self.current_iteration_source = self.current_iteration_target = 0
        self.domain_map = {
            "source": self.config.data_params.source,
            "target": self.config.data_params.target,
        }

        super(DEVAgent, self).__init__(config)

        # for MIM
        self.momentum_softmax_target = torchutils.MomentumSoftmax(
            self.num_class, m=len(self.get_attr("target", "train_loader"))
        )

        # init loss
        self.t = config.loss_params.T
        self.m = config.model_params.m

        if self.config.pretrained_exp_dir is None:
            self._init_memory_bank()

        # init statics
        self._init_labels() 
        
    def _define_task(self, config):
        # specify task
        self.cls = self.semi = self.tgt = self.ssl = False
        self.is_pseudo_src = self.is_pseudo_tgt = False
        for ls in config.loss_params.loss:
            self.cls = self.cls | ls.startswith("cls")
            self.tgt = self.tgt | ls.startswith("tgt")
            self.ssl = self.ssl | (ls.split("-")[0] not in ["cls", "tgt"])
            self.is_pseudo_tgt = self.is_pseudo_tgt | ls.startswith("tgt-pseudo")

        self.is_pseudo_tgt = self.is_pseudo_tgt | config.loss_params.pseudo

    def _init_labels(self):
        train_len_tgt = self.get_attr("target", "train_len")
        train_len_src = self.get_attr("source", "train_len")

        # labels for pseudo
        self.source_labels = (
            torch.zeros(train_len_src, dtype=torch.long).detach().cuda() - 1
        )
        for ind, _, lbl in self.get_attr("source", "train_loader"):
            lbl = torch.ones(1, dtype=torch.long).cuda() * lbl.cuda()
            self.source_labels[ind] = lbl
        self.predict_ordered_labels_pseudo_target = (
            torch.zeros(train_len_tgt, dtype=torch.long).detach().cuda() - 1
        )

    def _load_datasets(self):
        name = self.config.data_params.name
        num_workers = self.config.data_params.num_workers
        domain = self.domain_map
        image_size = self.config.data_params.image_size
        aug_src = self.config.data_params.aug_src
        aug_tgt = self.config.data_params.aug_tgt
        raw = "raw"

        self.num_class = datautils.get_class_num(
            f'data/splits/{name}/{domain["source"]}.txt'
        )
        self.class_map = datautils.get_class_map(
            f'data/splits/{name}/{domain["target"]}.txt'
        )

        batch_size_dict = {
            "test": self.config.optim_params.batch_size,
            "source": self.config.optim_params.batch_size_src,
            "target": self.config.optim_params.batch_size_tgt,
        }
        self.batch_size_dict = batch_size_dict

        # self-supervised Dataset
        for domain_name in ("source", "target"):
            aug_name = {"source": aug_src, "target": aug_tgt}[domain_name]

            # Training datasets
            train_dataset = datautils.create_dataset(
                name,
                domain[domain_name],
                suffix="",
                ret_index=True,
                image_transform=aug_name,
                use_mean_std=False,
                image_size=image_size,
            )

            train_loader = datautils.create_loader(
                train_dataset,
                batch_size_dict[domain_name],
                is_train=True,
                num_workers=num_workers,
            )
            train_init_loader = datautils.create_loader(
                train_dataset,
                batch_size_dict[domain_name],
                is_train=False,
                num_workers=num_workers,
            )
            train_labels = torch.from_numpy(train_dataset.labels).detach().cuda()

            self.set_attr(domain_name, "train_dataset", train_dataset)
            self.set_attr(domain_name, "train_ordered_labels", train_labels)
            self.set_attr(domain_name, "train_loader", train_loader)
            self.set_attr(domain_name, "train_init_loader", train_init_loader)
            self.set_attr(domain_name, "train_len", len(train_dataset))
            
        train_lbd_dataset_source = datautils.create_dataset(
            name,
            domain["source"],
            ret_index=True,
            image_transform=aug_src,
            image_size=image_size,
        )
        
        test_suffix = "test" if self.config.data_params.train_val_split else ""
        
        test_unl_dataset_target = datautils.create_dataset(
            name,
            domain["target"],
            suffix=test_suffix,
            ret_index=True,
            image_transform=raw,
            image_size=image_size,
        )

        self.train_lbd_loader_source = datautils.create_loader(
            train_lbd_dataset_source,
            batch_size_dict["source"],
            num_workers=num_workers,
        )
        self.test_unl_loader_target = datautils.create_loader(
            test_unl_dataset_target,
            batch_size_dict["test"],
            is_train=False,
            num_workers=num_workers,
        )

        self.logger.info(
            f"Dataset {name}, source {self.config.data_params.source}, target {self.config.data_params.target}"
        )

    def _create_model(self):
        version_grp = self.config.model_params.version.split("-")
        version = version_grp[-1]
        pretrained = "pretrain" in version_grp
        if pretrained:
            self.logger.info("Imagenet pretrained model used")
        out_dim = self.config.model_params.out_dim

        # backbone
        if "resnet" in version:
            if pretrained:
                backbone = get_model(version, pretrain=pretrained)
            else:
                backbone = get_model(version, pretrain=False)
            model = models.builder.FixCL(backbone, self.num_class, bottleneck_dim=out_dim, mlp=self.config.model_params.mlp, finetune=True)
            print(model)
        else:
            raise NotImplementedError

        # TODO: distributed
        model = nn.DataParallel(model, device_ids=self.gpu_devices)
        model = model.cuda()
        self.model = model

        self.criterion = CrossEntropyLabelSmooth(self.num_class).cuda()
        # self.criterion = nn.CrossEntropyLoss()
        
    def _create_optimizer(self):
        lr = self.config.optim_params.learning_rate
        momentum = self.config.optim_params.momentum
        weight_decay = self.config.optim_params.weight_decay
        conv_lr_ratio = self.config.optim_params.conv_lr_ratio
        
        parameters = []
        # batch_norm layer: no weight_decay
        params_bn, _ = torchutils.split_params_by_name(self.model, "bn")
        parameters.append({"params": params_bn, "weight_decay": 0.0})
        # conv layer: small lr
        _, params_conv = torchutils.split_params_by_name(self.model, ["fc", "bn", "head", "bottleneck"])
        if conv_lr_ratio:
            parameters[0]["lr"] = lr * conv_lr_ratio
            parameters.append({"params": params_conv, "lr": lr * conv_lr_ratio})
        else:
            parameters.append({"params": params_conv})
        # fc layer
        params_fc, _ = torchutils.split_params_by_name(self.model, ["fc", "bottleneck", "head"])
        parameters.append({"params": params_fc})
        # parameters = self.model.get_parameters()
        self.optim = torch.optim.SGD(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=self.config.optim_params.nesterov,
        )

        # lr schedular
        if self.config.optim_params.lr_decay_schedule:
            optim_stepLR = torch.optim.lr_scheduler.MultiStepLR(
                self.optim,
                milestones=self.config.optim_params.lr_decay_schedule,
                gamma=self.config.optim_params.lr_decay_rate,
            )
            self.lr_scheduler_list.append(optim_stepLR)

        if self.config.optim_params.decay:
            self.optim_iterdecayLR = torchutils.lr_scheduler_invLR(self.optim)
        
        
    def train_one_epoch(self):
        # train preparation
        self.model = self.model.train()
        
        loss_list = self.config.loss_params.loss
        loss_weight = self.config.loss_params.weight
        loss_warmup = self.config.loss_params.start
        loss_giveup = self.config.loss_params.end

        num_loss = len(loss_list)

        source_loader = self.get_attr("source", "train_loader")
        target_loader = self.get_attr("target", "train_loader")

        if self.config.steps_epoch is None:
            num_batches = max(len(source_loader), len(target_loader)) + 1
            self.logger.info(f"source loader batches: {len(source_loader)}")
            self.logger.info(f"target loader batches: {len(target_loader)}")
        else:
            num_batches = self.config.steps_epoch

        batch_time = AverageMeter('Time', ':6.3f')
        # load_time = AverageMeter('Load', ':6.3f')
        total_loss = AverageMeter('Loss', ':.4e')
        losses = []
        for loss_item in loss_list:
            losses.append(AverageMeter(loss_item, ':.4e'))
        acc = AverageMeter('Acc', ':6.2f')
        progress_list = [batch_time, total_loss]
        progress_list = progress_list+losses
        progress_list.append(acc)
        progress = ProgressMeter(
            num_batches,
            progress_list,
            prefix="Epoch: [{}]".format(self.current_epoch)
        )

        end = time.time()

        # TODO: Target - init target labels & target prototype
        if self.config.model_params.mix:
            k = self.num_class
            memory_bank_instance_target = self.get_attr("target", "memory_bank_wrapper").as_tensor()
            memory_bank_proto_target = self.get_attr("target", "memory_bank_proto")

            # if self.current_epoch <= 5:
            #     init_centroids = self.get_attr("source", "memory_bank_proto").as_tensor()
            # else:
            #     init_centroids = self.get_attr("target", "memory_bank_proto").as_tensor()
            
            init_centroids = self.get_attr("source", "memory_bank_proto").as_tensor()
            
            target_labels, target_cluster_centroids, target_cluster_phi = torch_kmeans(
                k_list=[k],
                data=memory_bank_instance_target,
                init_centroids=init_centroids,
                seed=self.current_epoch + self.current_iteration)
            
            tar_proto = memory_bank_proto_target.as_tensor()
            
            if self.current_epoch < 2:
                new_tar_proto = target_cluster_centroids[0]
            else:
                new_tar_proto = update_data_memory(tar_proto, target_cluster_centroids[0], m=self.m)
            memory_bank_proto_target.update(torch.arange(0, self.num_class, dtype=torch.long).cuda(), new_tar_proto)
            
            # if self.current_epoch < 2:
            #     new_tar_proto = target_cluster_centroids[0]
            #     memory_bank_proto_target.update(torch.arange(0, self.num_class, dtype=torch.long).cuda(), new_tar_proto)

        for batch_i in range(num_batches):
            # iteration over all source images
            if not batch_i % len(source_loader):
                source_iter = iter(source_loader)

            # iteration over all target images
            if not batch_i % len(target_loader):
                target_iter = iter(target_loader)

                if "tgt-condentmax" in self.config.loss_params.loss:
                    momentum_prob_target = (
                        self.momentum_softmax_target.softmax_vector.cuda()
                    )
                    self.momentum_softmax_target.reset()

            indices_source, images_source, labels_source = next(source_iter)
            indices_source = indices_source.cuda()
            images_source = images_source.cuda()
            labels_source = labels_source.cuda()
            
            indices_target, images_target, _ = next(target_iter)
            indices_target = indices_target.cuda()
            images_target = images_target.cuda()
            labels_target = target_labels.squeeze()[indices_target]
            
            feat_source, predict_source = self.model(images_source)
            feat_target, predict_target = self.model(images_target)
            feat_source = F.normalize(feat_source, dim=1)
            feat_target = F.normalize(feat_target, dim=1)
            
            # proto-each loss
            proto_source = self.get_attr("source", "memory_bank_proto") # K x C
            logits_source = torch.einsum('nc,kc->nk',[feat_source, proto_source.as_tensor()]) # N=bt_size, K=class_num

            logits_source /= self.t

            proto_target = self.get_attr("target", "memory_bank_proto")
            logits_target = torch.einsum('nc,kc->nk', [feat_target, proto_target.as_tensor()])
            logits_target /= self.t

            # I2M Loss
            mix_source = self.get_attr("source", "memory_bank_mix")
            mix_target = self.get_attr("target", "memory_bank_mix")
            
            logits_source_mix = torch.einsum('nc,kc->nk', [feat_source, mix_source.as_tensor()])
            logits_target_mix = torch.einsum('nc,kc->nk', [feat_target, mix_target.as_tensor()])
            logits_source_mix /= self.t
            logits_target_mix /= self.t
            
            # Compute Loss
            loss = torch.tensor(0).cuda()
            for ind, ls in enumerate(loss_list):
                if self.current_epoch < loss_warmup[ind] or self.current_epoch >= loss_giveup[ind]:
                    continue
                loss_part = torch.tensor(0).cuda()
                if ls == "cls-so":
                    loss_part = self.criterion(predict_source, labels_source)
                elif ls == "tgt-entmin":
                    loss_part = Entropy()(predict_target)
                elif ls == "tgt-condentmax":
                    bs = predict_target.size(0)
                    prob_target = F.softmax(predict_target, dim=1)
                    prob_mean_target = prob_target.sum(dim=0) / bs
                    
                    # update momentum
                    self.momentum_softmax_target.update(
                        prob_mean_target.cpu().detach(), bs
                    )
                    
                    # get momentum probablity
                    momentum_prob_target = (
                        self.momentum_softmax_target.softmax_vector.cuda()
                    )
                    
                    # compute loss
                    entropy_cond = - torch.sum(
                        prob_mean_target * torch.log(
                            momentum_prob_target + 1e-5
                        )
                    )
                    loss_part = - entropy_cond
                elif ls.split("-")[0] == "proto":
                    proto_source_loss = nn.CrossEntropyLoss()(logits_source, labels_source)
                    proto_target_loss = nn.CrossEntropyLoss()(logits_target, labels_target)
                    # proto_source_loss = CrossEntropyLabelSmooth(self.num_class)(logits_source, labels_source)
                    # proto_target_loss = CrossEntropyLabelSmooth(self.num_class)(logits_target, labels_target)
                    loss_part = (proto_source_loss + proto_target_loss) / 2
                elif ls.split("-")[0] == "I2M":
                    mix_source_loss = Entropy()(logits_source_mix)
                    mix_target_loss = Entropy()(logits_target_mix)
                    # mix_source_loss = CrossEntropyLabelSmooth(self.num_class)(logits_source_mix, labels_source)
                    # mix_target_loss = CrossEntropyLabelSmooth(self.num_class)(logits_target_mix, labels_target)
                    loss_part = (mix_source_loss + mix_target_loss) / 2
                    
                loss_part = loss_weight[ind] * loss_part
                losses[ind].update(loss_part.item(), images_source.size(0))
                loss = loss + loss_part
            total_loss.update(loss.item(), images_source.size(0))
            
            # Backpropagation
            self.optim.zero_grad()
            if len(loss_list) and loss != 0:
                loss.backward()
            self.optim.step()
            
            # update memory bank
            domain_m_dict = {
                "source": 0.8,
                "target": 0.2
            }
            
            # target instance
            memory_bank_target = self.get_attr("target", "memory_bank_wrapper").as_tensor()
            data_memory = torch.index_select(memory_bank_target, 0, indices_target)
            new_target_data = data_memory * self.m + (1 - self.m) * F.normalize(feat_target, dim=1)
            new_target_data = F.normalize(new_target_data, dim=1)
            self._update_memory_bank("target", indices_target, new_target_data)
            
            # source proto
            for idx in range(self.num_class):
                if len(feat_source[labels_source == idx]) == 0:
                    continue
                tmp_proto = feat_source[labels_source == idx].mean(0)
                idx = torch.ones(1, dtype=torch.int64) * idx
                idx = idx.cuda()
                proto_source = self.get_attr("source", "memory_bank_proto")
                old_proto = proto_source.at_idxs(idx)
                update_proto = update_data_memory(old_proto, tmp_proto.view(1, -1), m=self.m)
                proto_source.update(idx, update_proto)
            # target proto
            # for idx in range(self.num_class):
            #     if len(feat_target[labels_target == idx]) == 0:
            #             continue
            #     tmp_proto = feat_target[labels_target == idx].mean(0)
            #     idx = torch.ones(1, dtype=torch.int64) * idx
            #     idx = idx.cuda()
            #     memory_bank_proto_target = self.get_attr("target", "memory_bank_proto")
            #     old_proto = memory_bank_proto_target.at_idxs(idx)
            #     update_proto = update_data_memory(old_proto, tmp_proto.view(1, -1), m=self.m)
            #     memory_bank_proto_target.update(idx, update_proto)
            
            # mix proto
            proto_source = self.get_attr("source", "memory_bank_proto")
            proto_target = self.get_attr("target", "memory_bank_proto")
            mix_proto_source = self.get_attr("source", "memory_bank_mix")
            mix_proto_target = self.get_attr("target", "memory_bank_mix")
            update_mix_source = domain_m_dict["source"] * proto_source.as_tensor() + (1 - domain_m_dict["target"]) * proto_target.as_tensor()
            update_mix_target = domain_m_dict["target"] * proto_source.as_tensor() + domain_m_dict["source"] * proto_target.as_tensor()
            update_mix_source = F.normalize(update_mix_source, dim=1)
            update_mix_target = F.normalize(update_mix_target, dim=1)
            # update_mix_source = update_data_memory(mix_proto_source.as_tensor(), update_mix_source, m=self.m)
            # update_mix_target = update_data_memory(mix_proto_target.as_tensor(), update_mix_target, m=self.m)
            mix_proto_source.update(torch.arange(0, self.num_class, dtype=torch.long).cuda(), update_mix_source)
            mix_proto_target.update(torch.arange(0, self.num_class, dtype=torch.long).cuda(), update_mix_target)
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            print_freq = 10
            if batch_i % print_freq == 0:
                print(torchutils.get_lr(self.optim, g_id=-1))
                progress.display(batch_i)
               
    # Validate
    @torch.no_grad()
    def validate(self):
        self.model.eval()

        # Domain Adaptation
        if self.cls:
            self.current_val_metric = self.score(
                self.test_unl_loader_target,
                name=f"unlabeled {self.domain_map['target']}",
            )

        # update information
        self.current_val_iteration += 1
        if self.current_val_metric >= self.best_val_metric:
            self.best_val_metric = self.current_val_metric
            self.best_val_epoch = self.current_epoch
            self.iter_with_no_improv = 0
        else:
            self.iter_with_no_improv += 1
        self.val_acc.append(self.current_val_metric)

        self.clear_train_features()

    @torch.no_grad()
    def score(self, loader, name="test"):
        correct = 0
        size = 0
        epoch_loss = AverageMeter('Test Loss', ':.4e')
        error_indices = []
        confusion_matrix = torch.zeros(self.num_class, self.num_class, dtype=torch.long)
        pred_score = []
        pred_label = []
        label = []

        for batch_i, (indices, images, labels) in enumerate(loader):
            images = images.cuda()
            labels = labels.cuda()

            feat, output = self.model(images)
            prob = F.softmax(output, dim=-1)

            loss = self.criterion(output, labels)
            pred = torch.max(output, dim=1)[1]

            pred_label.extend(pred.cpu().tolist())
            label.extend(labels.cpu().tolist())
            if self.num_class == 2:
                pred_score.extend(prob[:, 1].cpu().tolist())

            correct += pred.eq(labels).sum().item()
            for t, p, ind in zip(labels, pred, indices):
                confusion_matrix[t.long(), p.long()] += 1
                if t != p:
                    error_indices.append((ind, p))
            size += pred.size(0)
            epoch_loss.update(loss, pred.size(0))

        acc = correct / size
        # self.summary_writer.add_scalars(
        #     "test/acc", {f"{name}": acc}, self.current_epoch
        # )
        # self.summary_writer.add_scalars(
        #     "test/loss", {f"{name}": epoch_loss.avg}, self.current_epoch
        # )
        self.logger.info(
            f"[Epoch {self.current_epoch} {name}] loss={epoch_loss.avg:.5f}, acc={correct}/{size}({100. * acc:.3f}%)"
        )

        return acc

    # Load & Save checkpoint

    def load_checkpoint(
        self,
        filename,
        checkpoint_dir=None,
        load_memory_bank=False,
        load_model=True,
        load_optim=False,
        load_epoch=False,
        load_cls=True,
    ):
        checkpoint_dir = checkpoint_dir or self.config.checkpoint_dir
        filename = os.path.join(checkpoint_dir, filename)
        try:
            self.logger.info(f"Loading checkpoint '{filename}'")
            checkpoint = torch.load(filename, map_location="cpu")

            if load_epoch:
                self.current_epoch = checkpoint["epoch"]
                for domain_name in ("source", "target"):
                    self.set_attr(
                        domain_name,
                        "current_iteration",
                        checkpoint[f"iteration_{domain_name}"],
                    )
                self.current_iteration = checkpoint["iteration"]
                self.current_val_iteration = checkpoint["val_iteration"]

            if load_model:
                model_state_dict = checkpoint["model_state_dict"]
                self.model.load_state_dict(model_state_dict)

            if load_cls and self.cls and "cls_state_dict" in checkpoint:
                cls_state_dict = checkpoint["cls_state_dict"]
                self.cls_head.load_state_dict(cls_state_dict)

            if load_optim:
                optim_state_dict = checkpoint["optim_state_dict"]
                self.optim.load_state_dict(optim_state_dict)

                lr_pretrained = self.optim.param_groups[0]["lr"]
                lr_config = self.config.optim_params.learning_rate

                # Change learning rate
                if not lr_pretrained == lr_config:
                    for param_group in self.optim.param_groups:
                        param_group["lr"] = self.config.optim_params.learning_rate

            self._init_memory_bank()
            if (
                load_memory_bank or self.config.model_params.load_memory_bank == False
            ):  # load memory_bank
                self._load_memory_bank(
                    {
                        "source": checkpoint["memory_bank_source"],
                        "target": checkpoint["memory_bank_target"],
                    }
                )

            self.logger.info(
                f"Checkpoint loaded successfully from '{filename}' at (epoch {checkpoint['epoch']}) at (iteration s:{checkpoint['iteration_source']} t:{checkpoint['iteration_target']}) with loss = {checkpoint['loss']}\nval acc = {checkpoint['val_acc']}\n"
            )

        except OSError as e:
            self.logger.info(f"Checkpoint doesnt exists: [{filename}]")
            raise e

    def save_checkpoint(self, filename="checkpoint.pth.tar"):
        out_dict = {
            "config": self.config,
            "model_state_dict": self.model.state_dict(),
            "optim_state_dict": self.optim.state_dict(),
            "memory_bank_source": self.get_attr("source", "memory_bank_wrapper"),
            "memory_bank_target": self.get_attr("target", "memory_bank_wrapper"),
            "epoch": self.current_epoch,
            "iteration": self.current_iteration,
            "iteration_source": self.get_attr("source", "current_iteration"),
            "iteration_target": self.get_attr("target", "current_iteration"),
            "val_iteration": self.current_val_iteration,
            "val_acc": np.array(self.val_acc),
            "val_metric": self.current_val_metric,
            "loss": self.current_loss,
            "train_loss": np.array(self.train_loss),
        }
        if self.cls:
            out_dict["cls_state_dict"] = self.cls_head.state_dict()
        # best according to source-to-target
        is_best = (
            self.current_val_metric == self.best_val_metric
        ) or not self.config.validate_freq
        torchutils.save_checkpoint(
            out_dict, is_best, filename=filename, folder=self.config.checkpoint_dir
        )
        self.copy_checkpoint()

    # compute train features
    @torch.no_grad()
    def compute_train_features(self):
        if self.is_features_computed:
            return
        else:
            self.is_features_computed = True
        self.model.eval()

        for domain in ("source", "target"):
            train_loader = self.get_attr(domain, "train_init_loader")
            features, y, idx = [], [], []
            tqdm_batch = tqdm(
                total=len(train_loader), desc=f"[Compute train features of {domain}]"
            )
            for batch_i, (indices, images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                feat, _ = self.model(images)
                feat = F.normalize(feat, dim=1)

                features.append(feat)
                y.append(labels)
                idx.append(indices)

                tqdm_batch.update()
            tqdm_batch.close()

            features = torch.cat(features)
            y = torch.cat(y)
            idx = torch.cat(idx).to(self.device)

            self.set_attr(domain, "train_features", features)
            self.set_attr(domain, "train_labels", y)
            self.set_attr(domain, "train_indices", idx)

    def clear_train_features(self):
        self.is_features_computed = False

    # Memory bank

    @torch.no_grad()
    def _init_memory_bank(self):
        out_dim = self.config.model_params.out_dim
        for domain_name in ("source", "target"):
            data_len = self.get_attr(domain_name, "train_len")
            memory_bank = MemoryBank(data_len, out_dim)
            memory_bank_mix = MemoryBank(self.num_class, out_dim)
            memory_bank_proto = MemoryBank(self.num_class, out_dim)
            if self.config.model_params.load_memory_bank:
                self.compute_train_features()
                idx = self.get_attr(domain_name, "train_indices")
                feat = self.get_attr(domain_name, "train_features")
                memory_bank.update(idx, feat)
                # TODO: Source - init source prototype
                if domain_name == "source" and self.config.model_params.mix:
                    labels = self.get_attr(domain_name, "train_ordered_labels")[idx]
                    for idx in range(self.num_class):
                        if len(feat[labels == idx]) == 0:
                            continue
                        idx = torch.ones(1, dtype=torch.int64) * idx
                        idx = idx.cuda()
                        old_proto = memory_bank_proto.at_idxs(idx)
                        new_proto = feat[labels == idx].mean(0).view(1,-1)
                        update_proto = new_proto
                        memory_bank_proto.update(idx, update_proto)
                # self.logger.info(
                #     f"Initialize memorybank-{domain_name} with pretrained output features"
                # )
                # save space
                if self.config.data_params.name in ["visda17", "domainnet"]:
                    delattr(self, f"train_indices_{domain_name}")
                    delattr(self, f"train_features_{domain_name}")
            if self.config.model_params.mix:
                self.set_attr(domain_name, "memory_bank_proto", memory_bank_proto)
            self.set_attr(domain_name, "memory_bank_wrapper", memory_bank)
            # TODO: Mix - init mix prototype
            self.set_attr(domain_name, "memory_bank_mix", memory_bank_mix)

    @torch.no_grad()
    def _update_memory_bank(self, domain_name, indices, new_data_memory):
        memory_bank_wrapper = self.get_attr(domain_name, "memory_bank_wrapper")
        memory_bank_wrapper.update(indices, new_data_memory)

    def _load_memory_bank(self, memory_bank_dict):
        """load memory bank from checkpoint

        Args:
            memory_bank_dict (dict): memory_bank dict of source and target domain
        """
        for domain_name in ("source", "target"):
            memory_bank = memory_bank_dict[domain_name]._bank.cuda()
            self.get_attr(domain_name, "memory_bank_wrapper")._bank = memory_bank
            # self.loss_fn.module.set_broadcast(domain_name, "memory_bank", memory_bank)
