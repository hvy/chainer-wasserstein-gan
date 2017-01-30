import argparse

from chainer import datasets, training, iterators, optimizers, optimizer
from chainer.training import updater, extensions

from models import Generator, Critic
from updater import WassersteinGANUpdater
from extensions import GeneratorSample
from iterators import RandomNoiseIterator, GaussianNoiseGenerator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=64)
    return parser.parse_args()


def train(args):
    nz = args.nz
    batch_size = args.batch_size
    epochs = args.epochs
    gpu = args.gpu

    # CIFAR-10 images in range [-1, 1] (tanh generator outputs)
    train, _ = datasets.get_cifar10(withlabel=False, ndim=3, scale=2)
    train -= 1.0
    train_iter = iterators.SerialIterator(train, batch_size)

    z_iter = RandomNoiseIterator(GaussianNoiseGenerator(0, 1, args.nz),
                                 batch_size)

    optimizer_generator = optimizers.RMSprop(lr=0.00005)
    optimizer_critic = optimizers.RMSprop(lr=0.00005)
    optimizer_generator.setup(Generator())
    optimizer_critic.setup(Critic())

    updater = WassersteinGANUpdater(
        iterator=train_iter,
        noise_iterator=z_iter,
        optimizer_generator=optimizer_generator,
        optimizer_critic=optimizer_critic,
        device=gpu)

    trainer = training.Trainer(updater, stop_trigger=(epochs, 'epoch'))
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.LogReport(trigger=(1, 'iteration')))
    trainer.extend(GeneratorSample(), trigger=(1, 'epoch'))
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'critic/loss',
            'critic/loss/real', 'critic/loss/fake', 'generator/loss']))
    trainer.run()


if __name__ == '__main__':
    args = parse_args()
    train(args)
