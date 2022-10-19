# hnvc

This is an experimental implementation of image compression based on neural networks.

## Requirements

- Python 3.10
- pipenv

## Setup

```bash
pipenv shell
pipenv install
```

## Training

```python
python train.py --path /path/to/images --logdir ./tb_logs --ckpt ./ckpt --epochs 40 --lr 1e-3 --minls 0.5 --maxls 0.9 --bs 48 --width 128 --height 128
```
