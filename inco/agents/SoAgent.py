import enum
import os
import sys
import time
from curses.ascii import BS
from tkinter import image_names

import models.builder
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from models import (CrossEntropyLabelSmooth, Entropy, MemoryBank, torch_kmeans,
                    update_data_memory)
from inco.models.loss import DomainAdversarialLoss
from sklearn import metrics
from tqdm import tqdm
from inco.utils import tsne
from utils import (AverageMeter, ProgressMeter, accuracy, datautils, get_model,
                   torchutils, tsne)

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


class SoAgent(BaseAgent):
    def __init__(self, config):
        self.config = config
        self._define_task(config)
        self.is_features_computed = False
        self.current_iteration_source = self.current_iteration_target = 0
        self.domain_map = {
            "source": self.config.data_params.source,
            "target": self.config.data_params.target,
        }

        super(SoAgent, self).__init__(config)

        # init loss
        self.t = config.loss_params.T
        self.m = config.model_params.m



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

        self.source_labels = (
            torch.zeros(train_len_src, dtype=torch.long).detach().cuda() - 1
        )
        for ind, _, lbl in self.get_attr("source", "train_loader"):
            lbl = torch.ones(1, dtype=torch.long).cuda() * lbl.cuda()
            self.source_labels[ind] = lbl
        
        # pseudo target labels: not use
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
            f'data/{name}/{domain["source"]}.txt'
        )
        self.logger.info(f"Class Num: {self.num_class}")
        self.class_map = datautils.get_class_map(
            f'data/{name}/{domain["target"]}.txt'
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
            model = models.builder.FixCL(backbone, self.num_class, project_dim=out_dim, mlp=self.config.model_params.mlp, finetune=True)
            print(model)
        else:
            raise NotImplementedError


        self.domain_discri = models.builder.DomainDiscriminator(in_feature=model.features_dim, hidden_size=1024).cuda()
        self.criterion = nn.CrossEntropyLoss()
        self.domain_adv = DomainAdversarialLoss(self.domain_discri).cuda()
        model = nn.DataParallel(model, device_ids=self.gpu_devices)
        model = model.cuda()
        self.model = model

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
        _, params_conv = torchutils.split_params_by_name(self.model, ["fc", "bn", "head", "bottleneck", "project_head"])
        if conv_lr_ratio:
            parameters[0]["lr"] = lr * conv_lr_ratio
            parameters.append({"params": params_conv, "lr": lr * conv_lr_ratio})
        else:
            parameters.append({"params": params_conv})
        # fc layer
        params_fc, _ = torchutils.split_params_by_name(self.model, ["fc", "bottleneck", "head", "project_head"])
        parameters.append({"params": params_fc})
 

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
            optim_cosLR = torchutils.lr_scheduler_cosLR(
                self.optim, self.config.num_epochs)
            self.lr_scheduler_list.append(optim_cosLR)
    
    def train_one_epoch(self):
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
        total_loss = AverageMeter('Loss', ':2.2f')
        losses = []
        for loss_item in loss_list:
            losses.append(AverageMeter(loss_item, ':2.2f'))
        progress_list = [batch_time, total_loss]
        progress_list = progress_list+losses
        progress = ProgressMeter(
            num_batches,
            progress_list,
            prefix="Epoch: [{}]".format(self.current_epoch)
        )

        if self.config.pretrained_exp_dir is None and self.current_epoch == 1:
            self._init_memory_bank()
        
        # train preparation
        self.model = self.model.train()
        self.domain_adv = self.domain_adv.train()
        accumulation_steps = self.config.optim_params.accumulation_step
        end = time.time()

        
        for batch_i in range(num_batches):
            # iteration over all source images
            if not batch_i % len(source_loader):
                source_iter = iter(source_loader)



            indices_source, images_source, labels_source = next(source_iter)
            indices_source = indices_source.cuda()
            images_source = images_source.cuda()
            labels_source = labels_source.cuda()
            
            _, _, predict_source = self.model(images_source)

            # Compute Loss
            loss = torch.tensor(0).cuda()
            for ind, ls in enumerate(loss_list):
                if self.current_epoch < loss_warmup[ind] or self.current_epoch >= loss_giveup[ind]:
                    continue
                loss_part = torch.tensor(0).cuda()
                if ls == "cls-so" and loss_weight[ind] != 0:
                    loss_part = self.criterion(predict_source, labels_source)
                loss_part = loss_weight[ind] * loss_part
                losses[ind].update(loss_part.item(), images_source.size(0))
                loss = loss + loss_part
            total_loss.update(loss.item(), images_source.size(0))
            
            # grad accumulation
            loss = loss / accumulation_steps

            if len(loss_list) and loss != 0:
                loss.backward()
            # Backpropagation
            if batch_i % accumulation_steps == 0:
                self.optim.step()
                self.optim.zero_grad()
                
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            print_freq = 10
            if batch_i % print_freq == 0:
                progress.display(batch_i)
        print(torchutils.get_lr(self.optim, g_id=-1))
    # Validate
    @torch.no_grad()
    def validate(self):
        self.model = self.model.eval()

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

            _, _, output = self.model(images)
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
        self.logger.info(
            f"[Epoch {self.current_epoch} {name}] loss={epoch_loss.avg:.5f}, acc={correct}/{size}({100. * acc:.3f}%)"
        )

        return acc

    def analysis(self):
        if self.config.phase == "analysis":
            source_loader = self.get_attr("source", "train_loader")
            target_loader = self.get_attr("target", "train_loader")
            self.load_checkpoint(filename="model_best.pth.tar")
            source_feature = self.collect_feature(data_loader=source_loader, device=self.device)
            target_feature = self.collect_feature(data_loader=target_loader, device=self.device)
            # plot t-SNE
            dir = os.path.join("/root/hotown/con_learning/images/source_only", self.config.exp_id)
            if not os.path.exists(dir):
                os.makedirs(dir)
            tSNE_filename = os.path.join(dir, 'TSNE.png')
            tsne.visualize(source_feature, target_feature, tSNE_filename)
            print("Saving t-SNE to", tSNE_filename)

    def collect_feature(self, data_loader: torch.utils.data.DataLoader,
                    device: torch.device, max_num_features=None) -> torch.Tensor:
        self.model.eval()
        all_features = []
        with torch.no_grad():
            for i, (_, images, target) in enumerate(data_loader):
                if max_num_features is not None and i >= max_num_features:
                    break
                images = images.to(device)
                feature, _, _ = self.model(images)
                feature = feature.cpu()
                all_features.append(feature)
        return torch.cat(all_features, dim=0)


    # Load & Save checkpoint

    def load_checkpoint(
        self,
        filename,
        checkpoint_dir=None,
        load_memory_bank=False,
        load_model=True,
        load_optim=False,
        load_epoch=False,
        load_cls=False,
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
                        
            self.logger.info(
                f"Checkpoint loaded successfully from '{filename}'\n"
            )

        except OSError as e:
            self.logger.info(f"Checkpoint doesnt exists: [{filename}]")
            raise e

    def save_checkpoint(self, filename="checkpoint.pth.tar"):
        out_dict = {
            "config": self.config,
            "model_state_dict": self.model.state_dict(), 
        }

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
        self.model = self.model.eval()

        for domain in ("source", "target"):
            train_loader = self.get_attr(domain, "train_init_loader")
            features, y, idx = [], [], []
            tqdm_batch = tqdm(
                total=len(train_loader), desc=f"[Compute train features of {domain}]"
            )
            for batch_i, (indices, images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                _, feat, _ = self.model(images)
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

