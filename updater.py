import numpy as np

from chainer import training, reporter, cuda
from chainer import Variable


class WassersteinGANUpdater(training.StandardUpdater):
    def __init__(self, *, iterator, noise_iterator, optimizer_generator,
                 optimizer_critic, device=-1):

        if optimizer_generator.target.name is None:
            optimizer_generator.target.name = 'generator'

        if optimizer_critic.target.name is None:
            optimizer_critic.target.name = 'critic'

        iterators = {'main': iterator, 'z': noise_iterator}
        optimizers = {'generator': optimizer_generator,
                      'critic': optimizer_critic}

        super().__init__(iterators, optimizers, device=device)

        if device >= 0:
            cuda.get_device(device).use()
            [optimizer.target.to_gpu() for optimizer in optimizers.values()]

        self.xp = cuda.cupy if device >= 0 else np

    @property
    def optimizer_generator(self):
        return self._optimizers['generator']

    @property
    def optimizer_critic(self):
        return self._optimizers['critic']

    @property
    def generator(self):
        return self._optimizers['generator'].target

    @property
    def critic(self):
        return self._optimizers['critic'].target

    @property
    def x(self):
        return self._iterators['main']

    @property
    def z(self):
        return self._iterators['z']

    def next_batch(self, iterator):
        batch = self.converter(iterator.next(), self.device)
        return Variable(batch)

    def sample(self):

        """Return a sample batch of images."""

        z = self.next_batch(self.z)
        x = self.generator(z, test=True)

        # [-1, 1] -> [0, 1]
        x += 1.0
        x /= 2

        return x

    def update_core(self):

        def _update(optimizer, loss):
            optimizer.target.cleargrads()
            loss.backward()
            optimizer.update()

        # Update critic 5 times
        for _ in range(5):
            # Clamp critic parameters
            self.critic.clamp()

            # Real images
            x_real = self.next_batch(self.x)
            y_real = self.critic(x_real)
            y_real.grad = self.xp.ones_like(y_real.data)
            _update(self.optimizer_critic, y_real)

            # Fake images
            z = self.next_batch(self.z)
            x_fake = self.generator(z)
            y_fake = self.critic(x_fake)
            y_fake.grad = -1 * self.xp.ones_like(y_fake.data)
            _update(self.optimizer_critic, y_fake)

            reporter.report({
                'critic/loss/real': y_real,
                'critic/loss/fake': y_fake,
                'critic/loss': y_real - y_fake
            })

        # Update generator 1 time
        z = self.next_batch(self.z)
        x_fake = self.generator(z)
        y_fake = self.critic(x_fake)
        y_fake.grad = self.xp.ones_like(y_fake.data)
        _update(self.optimizer_generator, y_fake)

        reporter.report({'generator/loss': y_fake})
