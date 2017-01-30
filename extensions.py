import os
import math

from matplotlib import pyplot as plt

from chainer import training, reporter, cuda
from chainer.training import extension


def save_ims(filename, ims, dpi=100):
    n, c, w, h = ims.shape
    x_plots = math.ceil(math.sqrt(n))
    y_plots = x_plots if n % x_plots == 0 else x_plots - 1
    plt.figure(figsize=(w*x_plots/dpi, h*y_plots/dpi), dpi=dpi)

    for i, im in enumerate(ims):
        plt.subplot(y_plots, x_plots, i+1)

        if c == 1:
            plt.imshow(im[0])
        else:
            plt.imshow(im.transpose((1, 2, 0)))

        plt.axis('off')
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.gray()
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0,
                            hspace=0)

    plt.savefig(filename, dpi=dpi*2, facecolor='black')
    plt.clf()
    plt.close()


class GeneratorSample(extension.Extension):
    def __init__(self, dirname='sample', sample_format='png'):
        self._dirname = dirname
        self._sample_format = sample_format

    def __call__(self, trainer):
        dirname = os.path.join(trainer.out, self._dirname)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        x = self.sample(trainer)

        filename = '{}.{}'.format(trainer.updater.epoch,
                                  self._sample_format)
        filename = os.path.join(dirname, filename)
        save_ims(filename, x)

    def sample(self, trainer):
        x = trainer.updater.sample()
        x = x.data
        if cuda.get_array_module(x) == cuda.cupy:
            x = cuda.to_cpu(x)
        return x
