""" Script for creating a Convolutional Neural Network
    using the torch.nn Module.
    Uses the higher level apis => torch.nn and torch.nn.functional
"""

import numpy as np
import argparse
import torch as th
import torchvision as tv
from functools import reduce


def setup_data(batch_size, num_workers, download=False):
    """
    setup the CIFAR-10 dataset for training the CNN
    :param batch_size: batch_size for sgd
    :param num_workers: num_readers for data reading
    :param download: Boolean for whether to download the data
    :return: classes, trainloader, testloader => training and testing data loaders
    """
    # data setup:
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    transforms = tv.transforms.ToTensor()

    trainset = tv.datasets.CIFAR10(root="data/cifar-10",
                                   transform=transforms,
                                   download=download)
    trainloader = th.utils.data.DataLoader(trainset, batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=num_workers)

    testset = tv.datasets.CIFAR10(root="data/cifar-10",
                                  transform=transforms, train=False,
                                  download=False)
    testloader = th.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=num_workers)

    return classes, trainloader, testloader


class CNN(th.nn.Module):
    """ Convolutional Neural Network implementation using torch's HL api
        Works with CIFAR-10 dataset.
        input => 32 x 32 x 3
        output => 10 (number of classes)
    """

    def __init__(self):
        """ Constructor """
        super(CNN, self).__init__()

        # define the dimensions of the Cifar-10 dataset
        self.dim = 32
        self.n_channels = 3
        self.n_classes = 10  # number of output classes

        # define the submodules to be used by the network
        # conv-layers
        self.conv1 = th.nn.Conv2d(self.n_channels, 16, 5, padding=2)
        self.conv2 = th.nn.Conv2d(16, 32, 5, padding=2)
        self.conv3 = th.nn.Conv2d(32, 64, 3, padding=1)

        # fully-connected layers
        self.fc_1 = th.nn.Linear(64 * 4 * 4, 256)
        self.fc_2 = th.nn.Linear(256, 64)
        self.fc_3 = th.nn.Linear(64, self.n_classes, bias=False)

    def forward(self, x):
        """
        forward pass of the network
        :param x: input images of the network [size = (B x 3 x 32 x 32)]
        :return: out [class probabilities = (B x 10)]
        """
        import torch.nn.functional as f

        # Conv - Max pooling over a (2, 2) window
        x = f.max_pool2d(f.relu(self.conv1(x)), 2)
        x = f.max_pool2d(f.relu(self.conv2(x)), 2)
        x = f.max_pool2d(f.relu(self.conv3(x)), 2)

        # Fully connected layers at the end
        x = x.view(-1, self._num_flat_features(x))  # flatten the output
        x = f.relu(self.fc_1(x))
        x = f.relu(self.fc_2(x))
        x = self.fc_3(x)

        # return raw class probabilities
        return x

    @staticmethod
    def _num_flat_features(x):
        """
        obtain the number of flat features of a tensor (Possibly 4D)
        :param x: input tensor
        :return: num_flat => number of flat features
        """

        size = x.size()[1:]  # all dims except the first one
        num_flat = reduce(lambda a, b: a * b, size)

        # return num_flat features
        return num_flat


def parse_arguments():
    """
    Command line argument parser function
    :return: args => parsed arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", action="store", type=int, default=32,
                        help="Batch size for Stochastic Gradient Descent")
    parser.add_argument("--num_readers", action="store", type=int, default=3,
                        help="Number of parallel data readers")
    parser.add_argument("--download_data", action="store", type=bool, default=False,
                        help="Boolean for downloading data")

    args = parser.parse_args()
    return args


def train_network(network, train_data, conv_threshold=1e-4,
                  max_epochs=100, learning_rate=0.1, feed_back_factor=10):
    """
    train the network using the provided data
    :param network: CNN object
    :param train_data: dataloader for training dataset
    :param conv_threshold: threshold for convergence
    :param max_epochs: max number of epochs
    :param learning_rate: learning rate for Stochastic Gradient Descent
    :param feed_back_factor: number of losses to be printed per epoch
    :return:
    """
    loss_delta = float(np.inf)
    prev_loss = 0
    epoch = 1

    criterion = th.nn.CrossEntropyLoss()
    optim = th.optim.SGD(network.parameters(), lr=learning_rate)

    print("Starting the training process ...")
    # run the training loop
    while loss_delta > conv_threshold and epoch <= max_epochs:

        print("\nEpoch: %d" % epoch)

        total_batches = len(iter(train_data))
        losses = []  # initialize to empty list

        # loop over all the minibatches
        for (i, data) in enumerate(train_data, 1):

            # extract images and labels from the data
            images, labels = data

            # run the network computations:
            outs = network(images)
            loss = criterion(outs, labels)

            # clear gradients
            optim.zero_grad()

            # backpropagate and update weights
            loss.backward()
            optim.step()

            losses.append(loss.item())

            # print loss for every_feedback value
            if i % int(total_batches / feed_back_factor) == 0 or i == 1:
                print("Current Minibatch: %d  Loss: %.3f" % (i, loss.item()))

        avg_loss = np.mean(losses)

        print("Average Loss: %.3f" % avg_loss)

        # reassign prev_loss and calculate loss_delta
        loss_delta = np.abs(avg_loss - prev_loss)
        prev_loss = avg_loss

        epoch += 1

    print("Training complete ...")


def main(args):
    """
    Main function of the script
    :param args: parsed command line arguments
    :return: None
    """
    network = CNN()

    # Move the network computations to GPU if it is available
    if th.cuda.is_available():
        network = network.cuda()

    classes, train, test = setup_data(
                                batch_size=args.batch_size,
                                num_workers=args.num_readers
                            )

    train_network(network, train)


if __name__ == '__main__':
    main(parse_arguments())
