# Spike-Element-Wise-ResNet

This repository contains the codes for the paper [Deep Residual Learning in Spiking Neural Networks](https://arxiv.org/abs/2102.04159). We used a identical seed during training, and we can ensure that the user can get almost the same accuracy when using our codes to train. Some of the trained models for ImageNet are available at: https://figshare.com/articles/software/Spike-Element-Wise-ResNet/14752998

## Dependency

The origin codes uses a specific SpikingJelly. To maximize reproducibility, the user can download the latest SpikingJelly and rollback to the version that we used to train:

```bash
git clone https://github.com/fangwei123456/spikingjelly.git
cd spikingjelly
git reset --hard 2958519df84ad77c316c6e6fbfac96fb2e5f59a3
python setup.py install
```

Here is the commit information:

```bash
commit 2958519df84ad77c316c6e6fbfac96fb2e5f59a3
Author: fangwei123456 <fangwei123456@pku.edu.cn>
Date:   Wed May 12 18:05:33 2021 +0800
```

# Running Examples

### Train on ImageNet

```bash
cd imagenet
```

Train the Spiking ResNet-18 with zero-init:

```bash
python train.py --cos_lr_T 320 --model spiking_resnet18 -b 32 --output-dir ./logs --tb --print-freq 4096 --amp --cache-dataset --T 4 --lr 0.1 --epoch 320 --data-path /raid/wfang/imagenet --device cuda:0 --zero_init_residual
```

Train the SEW ResNet-18:

```bash
python train.py --cos_lr_T 320 --model sew_resnet18 -b 32 --output-dir ./logs --tb --print-freq 4096 --amp --cache-dataset --connect_f ADD --T 4 --lr 0.1 --epoch 320 --data-path /raid/wfang/imagenet --device cuda:0
```

Train the SEW ResNet-18 with 8 GPUs:

```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --cos_lr_T 320 --model sew_resnet18 -b 32 --output-dir ./logs --tb --print-freq 4096 --amp --cache-dataset --connect_f ADD --T 4 --lr 0.1 --epoch 320 --data-path /raid/wfang/imagenet
```

### Train on DVS Gesture

```bash
cd dvsgesture
```

Train the Spiking ResNet:

```bash
python train.py --tb --amp --output-dir ./logs --model SpikingResNet --device cuda:0 --lr-step-size 64 --epoch 192 --T_train 12 --T 16 --data-path /raid/wfang/datasets/DVS128Gesture
```

Train the SEW ResNet:

```bash
python train.py --tb --amp --output-dir ./logs --model SEWResNet --connect_f ADD --device cuda:0 --lr-step-size 64 --epoch 192 --T_train 12 --T 16 --data-path /raid/wfang/datasets/DVS128Gesture
```

Train the Plain Net:

```
python train.py --tb --amp --output-dir ./logs --model PlainNet --device cuda:0 --lr-step-size 64 --epoch 192 --T_train 12 --T 16 --data-path /raid/wfang/datasets/DVS128Gesture
```

You can also use multi GPUs to train the network. But it maybe unnecessary because using 1 GPU is fast enough.
