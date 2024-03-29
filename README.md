# InCo: Intermediate Prototype Contrast for Unsupervised Domain Adaptation

This repository contains the author's implementation in Pytorch for InCo(Intermediate Prototype Contrast for Unsupervised Domain Adaptation).

<!-- ## Overview

Architecture of InCo

![Architecture of InCo]() -->

## Requirements

```bash
conda env create -f environment.yml
pip install -r requirements.txt
```

## Training

- Download or soft-link the dataset under `data` folder (Supported datasets are ImageCLEF, Office-31, Office-Home)
- Run following command to train the InCo:

```
python inco/run.py --config config/${dataset}$/${task}$.json
python inco/run.py --config config/office/A-D-all.json
```

## Acknowledgement

This code is built on [PCS](https://github.com/zhengzangw/PCS-FUDA) and [TLlib](https://github.com/thuml/Transfer-Learning-Library). Thanks to the authors for sharing their codes.
