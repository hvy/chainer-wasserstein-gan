# Wasserstein GAN

Chainer implementation of the Wesserstein GAN by Martin Arjovsky et al. Note that this is not the official implementation. The official implementation is https://github.com/martinarjovsky/WassersteinGAN.

## Run

Train the models with CIFAR-10. Randomly sampled images will be saved under a subdirectory called `result` automatically created when training.

```bash
python train.py --batch-size 64 --epochs 100 --gpu 1
```

## Sample

Plotting the estimates with CIFAR-10.

![](images/loss.png)
