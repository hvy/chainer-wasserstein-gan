import chainer
from chainer import functions as F
from chainer import links as L


class Generator(chainer.Chain):

    """(batch_size, n_z) -> (batch_size, 3, 32, 32)"""

    def __init__(self):
        super().__init__(
            dc1=L.Deconvolution2D(None, 256, 4, stride=1, pad=0, nobias=True),
            dc2=L.Deconvolution2D(256, 128, 4, stride=2, pad=1, nobias=True),
            dc3=L.Deconvolution2D(128, 64, 4, stride=2, pad=1, nobias=True),
            dc4=L.Deconvolution2D(64, 3, 4, stride=2, pad=1, nobias=True),
            bn_dc1=L.BatchNormalization(256),
            bn_dc2=L.BatchNormalization(128),
            bn_dc3=L.BatchNormalization(64)
        )

    def __call__(self, z, test=False):
        h = F.reshape(z, (z.shape[0], -1, 1, 1))
        h = F.relu(self.bn_dc1(self.dc1(h), test=test))
        h = F.relu(self.bn_dc2(self.dc2(h), test=test))
        h = F.relu(self.bn_dc3(self.dc3(h), test=test))
        h = F.tanh(self.dc4(h))
        return h


class Critic(chainer.Chain):

    """(batch_size, 3, 32, 32) -> ()"""

    def __init__(self):
        super().__init__(
            c0=L.Convolution2D(3, 64, 4, stride=2, pad=1, nobias=True),
            c1=L.Convolution2D(64, 128, 4, stride=2, pad=1, nobias=True),
            c2=L.Convolution2D(128, 256, 4, stride=2, pad=1, nobias=True),
            c3=L.Convolution2D(256, 1, 4, stride=1, pad=0, nobias=True),
            bn_c1=L.BatchNormalization(128),
            bn_c2=L.BatchNormalization(256)
        )

    def clamp(self, lower=-0.01, upper=0.01):

        """Clamp all parameters, including the batch normalization
        parameters."""

        for params in self.params():
            params_clipped = F.clip(params, lower, upper)
            params.data = params_clipped.data

    def __call__(self, x, test=False):
        h = F.leaky_relu(self.c0(x))
        h = F.leaky_relu(self.bn_c1(self.c1(h), test=test))
        h = F.leaky_relu(self.bn_c2(self.c2(h), test=test))
        h = self.c3(h)
        h = F.sum(h) / h.size  # Mean
        return h
