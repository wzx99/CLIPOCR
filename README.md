# Symmetrical Linguistic Feature Distillation with CLIP for Scene Text Recognition

This is a pytorch implementation for paper [CLIPOCR](https://arxiv.org/abs/2310.04999)

## Installation

### Requirements

- Python==3.8.12
- Pytorch==1.11.0
- CUDA==11.3

```bash
conda create --name CLIPOCR python=3.8.12 -y
conda activate CLIPOCR
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

### Datasets

Download the datasets to the "data" folder following [parseq](https://github.com/baudm/parseq).
The structure of data folder as below.

  ```bash
  data
  ├── test
  │   ├── CUTE80
  │   ├── IC13_857
  │   ├── IC15_1811
  │   ├── IIIT5k
  │   ├── SVT
  │   └── SVTP
  ├── train
  │   └── synth
  │       ├── MJ
  │       │   ├── train
  │       │   ├── test
  │       │   └── valid
  │       └── ST
  ```

## Training

```bash
python train.py trainer.gpus=2 ckpt_name=clipocr_synth dataset=synth model=clipocr model.batch_size=160 trainer.val_check_interval=1.0 trainer.max_epochs=5 model.lr=0.0014
```

## Testing
Pretrained model is available in [here](https://drive.google.com/drive/folders/1IYmXlwFrnpgizioGd7l_pk4zlMEgK1_U?usp=share_link).
```bash
python test.py ckpt/clipocr_synth/run/checkpoints --data_root data
```
