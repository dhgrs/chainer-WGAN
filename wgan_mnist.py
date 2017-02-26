# coding: UTF-8
import argparse
import os

import numpy as np
from PIL import Image
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.dataset import iterator as iterator_module
from chainer.training import extensions
from chainer.dataset import convert


class Generator(chainer.Chain):
    def __init__(self, ):
        super(Generator, self).__init__(
            fc1=L.Linear(None, 800),
            fc2=L.Linear(None, 28 * 28)
            )

    def __call__(self, z, test=False):
        h = F.relu(self.fc1(z))
        y = F.reshape(F.sigmoid(self.fc2(h)), (-1, 1, 28, 28))
        return y


class Critic(chainer.Chain):
    def __init__(self):
        super(Critic, self).__init__(
            fc1=L.Linear(None, 800),
            fc2=L.Linear(None, 28 * 28)
            )

    def __call__(self, x, test=False):
        batchsize = x.shape[0]
        h = F.relu(self.fc1(x))
        y = F.sum(self.fc2(h)) / batchsize
        return y


class WGANUpdater(training.StandardUpdater):
    def __init__(self, iterator, generator, critic,
                 n_c, opt_g, opt_c, device):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator
        self.generator = generator
        self.critic = critic
        self.n_c = n_c
        self._optimizers = {'generator': opt_g, 'critic': opt_c}
        self.device = device
        self.converter = convert.concat_examples
        self.iteration = 0

    def update_core(self):
        # read data
        batch = self._iterators['main'].next()
        images = self.converter(batch, self.device)
        batchsize = images.shape[0]
        H, W = images.shape[2], images.shape[3]
        xp = chainer.cuda.get_array_module(images)

        # Step1 Generate
        z = xp.random.normal(
            size=(batchsize, 1, H // 4, W // 4)).astype(xp.float32)
        generated = self.generator(z)

        # Step2 Critic
        y_real = self.critic(images)
        y_fake = self.critic(generated)

        # Step3 Compute loss
        wasserstein_distance = y_real - y_fake
        loss_critic = -wasserstein_distance
        loss_generator = -y_fake

        # Step4 Update critic
        self.critic.cleargrads()
        loss_critic.backward()
        self._optimizers['critic'].update()

        # Step5 Update generator
        if self.iteration < 2500 and self.iteration % 100 == 0:
            self.generator.cleargrads()
            loss_generator.backward()
            self._optimizers['generator'].update()

        if self.iteration > 2500 or self.iteration % self.n_c == 0:
            self.generator.cleargrads()
            loss_generator.backward()
            self._optimizers['generator'].update()

        # Step6 Report
        chainer.reporter.report({
            'loss/generator': loss_generator, 'loss/critic': loss_critic,
            'wasserstein distance': wasserstein_distance})


class WeightClipping(object):
    name = 'WeightClipping'

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, opt):
        for param in opt.target.params():
            xp = chainer.cuda.get_array_module(param.data)
            param.data = xp.clip(param.data, -self.threshold, self.threshold)


def main():
    parser = argparse.ArgumentParser(description='WGAN')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()

    # Networks
    generator = Generator()
    critic = Critic()
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        generator.to_gpu()
        critic.to_gpu()

    # Optimizers
    opt_g = chainer.optimizers.RMSprop(5e-5)
    opt_g.setup(generator)
    opt_g.add_hook(chainer.optimizer.GradientClipping(1))

    opt_c = chainer.optimizers.RMSprop(5e-5)
    opt_c.setup(critic)
    opt_c.add_hook(chainer.optimizer.GradientClipping(1))
    opt_c.add_hook(WeightClipping(0.01))

    # Dataset
    train, _ = chainer.datasets.get_mnist(withlabel=False, ndim=3)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    # Trainer
    updater = WGANUpdater(train_iter, generator, critic, 5,
                          opt_g, opt_c, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Generate sample images
    def out_generated_image(generator, H, W, rows, cols, dst):
        @chainer.training.make_extension()
        def make_image(trainer):
            n_images = rows * cols
            xp = generator.xp
            z = xp.random.randn(n_images, 1, H // 4, W // 4).astype(xp.float32)
            x = generator(z, test=True)
            x = chainer.cuda.to_cpu(x.data)

            x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
            channels = x.shape[1]
            x = x.reshape((rows, cols, channels, H, W))
            x = x.transpose(0, 3, 1, 4, 2)
            x = x.reshape((rows * H, cols * W, channels))
            x = np.squeeze(x)

            preview_dir = '{}/preview'.format(dst)
            preview_path = preview_dir +\
                '/image{:0>5}.png'.format(trainer.updater.epoch)
            if not os.path.exists(preview_dir):
                os.makedirs(preview_dir)
            Image.fromarray(x).save(preview_path)
        return make_image

    # Extensions
    trainer.extend(extensions.dump_graph('wasserstein distance'))
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(
        extensions.PlotReport(['wasserstein distance'],
                              'epoch', file_name='distance.png'))
    trainer.extend(
        extensions.PlotReport(
            ['loss/generator'], 'epoch', file_name='loss.png'))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'wasserstein distance', 'loss/generator', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())
    trainer.extend(out_generated_image(generator, 28, 28, 5, 5, args.out),
                   trigger=(1, 'epoch'))
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    # Run
    trainer.run()

if __name__ == '__main__':
    main()
