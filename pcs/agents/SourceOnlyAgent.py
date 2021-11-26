import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from utils import (torchutils, datautils, AverageMeter, utils)
from tqdm import tqdm
from . import BaseAgent




class SourceOnlyAgent(BaseAgent):
    def __init__(self, config):
        self.config = config
        self._define_task(config)
        self.current_iteration_source = self.current_iteration_target = 0
        self.domain_map = {
            "source": self.config.data_params.source,
            "target": self.config.data_params.target
        }
        
        super(BaseAgent, self).__init__(config)

        # init loss
        loss_ce = nn.CrossEntropyLoss()
        loss_ce = nn.DataParallel(loss_ce, device_ids=self.gpu_devices).cuda()
        self.loss_ce = loss_ce

    def _define_task(self, config):
        raise NotImplementedError

    def _load_datasets(self):
        name = self.config.data_params.name
        num_workers = self.config.data_params.num_workers
        # fewshot = 
        domain = self.domain_map
        image_size = self.config.data_params.image_size
        aug_src = self.config.data_params.aug_src
        # aug_tgt =
        raw = "raw"

        self.num_class = datautils.get_class_num(
            f'data/splits/{name}/{domain["source"]}.txt'
        )
        # self.class_map = datautils.get_class_map(
        #     f'data/split/{name}/{domain["target"]}.txt'
        # )
        batch_size_dict = {
            "test": self.config.optim_params.batch_size,
            "source": self.config.optim_params.batch_size_src,
            "target": self.config.optim_params.batch_size_tgt,
            "labeled": self.config.optim_params.batch_size_lbd,
        }
        self.batch_size_dict = batch_size_dict

        aug_name = aug_src
        domain_name = "source"
        train_dataset = datautils.create_dataset(
            name,
            domain[domain_name],
            suffix="",
            ret_index=True,
            image_transform=aug_name,
            use_mean_std=False,
            image_size=image_size
        )

        train_loader = datautils.create_loader(
            train_dataset,
            batch_size_dict[domain_name],
            is_train=True,
            num_workers=num_workers
        )
        train_init_loader = datautils.create_loader(
            train_dataset,
            batch_size_dict[domain_name],
            is_train=False,
            num_workers=num_workers
        )
        train_labels = torch.from_numpy(train_dataset.labels)
        