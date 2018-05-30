""" Script for training a dense neural network on Synthetic
    data. Implementation does not use the PyTorch's layers
    or Functional API.
    Uses Autograd Module for better understanding
"""
import argparse
import numpy as np
import torch as th


def generate_random_data_sample(sample_size, feature_dim, num_classes, bias=5):
    """
    generates synthetic data for classification
    :param sample_size: number of samples to be generated
    :param feature_dim: input features' dimension
    :param num_classes: number of classes for classification
    :param bias: how far should the data be scattered in clusters
    :return: x,y => synthetic data
    """
    # generate random classes
    y = np.random.randint(size=(sample_size, 1), low=0, high=num_classes)

    # Make sure that the data is separable
    # bias = 3 and scaling using (y + 1) ensures clustering
    x = (np.random.randn(sample_size, feature_dim) + bias) * (y + 1)

    # convert x to single precision float datatype
    x = x.astype(np.float32)

    # mean normalize x:
    x = (x - x.mean()) / x.std()

    # return generated data
    return x, y


class NeuralNetwork:
    """ Neural network implementing forward and backward passes
        using only the default autograd package of PyTorch
    """

    def __generate_weights(self):
        """ private helper for generating network weights """

        weights = []  # initialize with empty list
        biases = []

        # shorthand for generating weights
        create_weight = lambda i, o: th.randn(o, i, dtype=th.float32,
                                              requires_grad=True)
        create_bias = lambda o: th.zeros(o, 1, dtype=th.float32,
                                         requires_grad=True)

        if self.depth > 0:
            # add weight corresponding to the first layer:
            weights.append(create_weight(self.inp_dim, self.widths[0]))
            biases.append(create_bias(self.widths[0]))

            # create the intermediate weights for the neural network
            for lay in range(1, self.depth):
                weights.append(create_weight(self.widths[lay - 1], self.widths[lay]))
                biases.append(create_bias(self.widths[lay]))

            # add the last weight to the weights list
            weights.append(create_weight(self.widths[-1], self.n_classes))
            biases.append(create_bias(self.n_classes))

        else:
            # logistic regression case
            weights.append(create_weight(self.inp_dim, self.n_classes))
            biases.append(create_bias(self.n_classes))

        # len of weights and biases must be equal
        assert len(weights) == len(biases)

        # return the generated list of weight tensors
        return weights, biases

    def __clear_gradients(self):
        """
        set the gradients of all the weights and biases to zero
        :return: None
        """

        for weight, bias in zip(self.weights, self.biases):
            weight.grad = th.zeros_like(weight)
            bias.grad = th.zeros_like(bias)

    def __init__(self, f_dim, n_classes, depth=3,
                 widths=(32, 32, 32), activation_fn=th.nn.functional.leaky_relu,
                 learning_rate=0.01):
        """
        constructor for the class
        :param f_dim: input features dimensionality
        :param n_classes: number of output classes
        :param depth: depth of the network (not including output and input)
                      [can be 0, but not -ve]
        :param widths: list of layerwidths [len(widths) == depth]
        :param activation_fn: activation function of the network
        :param learning_rate: learning rate for gradient descent
        """

        assert depth >= 0, "depth cannot be negative"
        assert len(widths) == depth, "len(widths) and depth don't match"

        # initialize class state:
        self.inp_dim = f_dim
        self.n_classes = n_classes
        self.depth = depth
        self.widths = widths
        self.act_fn = activation_fn
        self.lr = learning_rate

        # create weights for the network
        self.weights, self.biases = self.__generate_weights()

        # create a state var for current loss (cache for backpropagation)
        self.loss = None

    def forward(self, x):
        """
        forward pass of the network
        :param x: input to the network [dim(x) == self.inp_dim]
        :return: y => un_normalized predictions
        """
        assert x.shape[-1] == self.inp_dim, "input shape is not compatible " + str(x.shape)

        # implementation is very simple:
        y = x.t()  # start with transpose of x
        for weight, bias in zip(self.weights, self.biases):
            y = self.act_fn((weight @ y) + bias)

        # return the transpose of computed predictions
        return y.t()

    def calc_loss(self, x_in, target):
        """
        calculate loss (cross entropy) of current input wrt. the provided target
        *** Note that this doesn't cache in the predictions
        calls forward internally. Do not call redundantly
        :param x_in: input to neural network
        :param target: target classes [dim(target) == self.n_classes]
        :return: loss => cross-entropy loss value
        """

        # need to flatten targets
        target = target.view(target.numel())

        # obtain raw predictions
        preds = self.forward(x_in)

        self.loss = th.nn.functional.cross_entropy(preds, target)

        return self.loss.item()

    def backward(self):
        """
        performs backpropagation and
        adjusts the weights of the neural network
        using gradient descent
        :return: None
        """
        if self.loss is None:
            print("loss not computed. No backpropagation will take place")

        else:
            self.__clear_gradients()

            # compute the gradient of loss wrt all the weights.
            self.loss.backward()

            # update the weights and biases using the computed gradients
            for weight, bias in zip(self.weights, self.biases):
                weight.data -= (self.lr * weight.grad.data)
                bias.data -= (self.lr * bias.grad.data)


def calc_accuracy(network, x, y):
    """
    calculate accuracy of the network (% classification performance)
    on the given data
    :param network:
    :param x: input features
    :param y: target labels
    :return: acc => % accuracy
    """
    preds = network.forward(x).data.numpy()
    y = y.numpy()
    correct = np.equal(np.argmax(preds, axis=-1), y.flatten())
    return np.mean(correct) * 100


def train_network(network, dat_x, dat_y, val_x=None, val_y=None,
                  test_x=None, test_y=None,
                  conv_thresh=1e-4, max_epochs=500, cv=False):
    """
    train the Neural network using the given synthetic data
    :param network: Neural Network object
    :param dat_x: input features
    :param dat_y: target labels
    :param val_x: validation features
    :param val_y: validation target labels
    :param test_x: test features
    :param test_y: test target labels
    :param conv_thresh: convergence threshold
    :param max_epochs: maximum epochs for training
    :param cv: Perform cross validation per epoch or Not
    :return: None
    """
    # convert dat_x, dat_y to tensors
    x = th.from_numpy(dat_x)
    y = th.from_numpy(dat_y)

    if val_x is not None and val_y is not None:
        v_x = th.from_numpy(val_x)
        v_y = th.from_numpy(val_y)

    if test_x is not None and test_y is not None:
        t_x = th.from_numpy(test_x)
        t_y = th.from_numpy(test_y)

    # initialize convergence parameters
    loss_delta = float(np.inf)
    prev_loss = 0
    epoch = 1

    # start the training loop
    while loss_delta >= conv_thresh and epoch <= max_epochs:
        # compute the forward loss of the network
        curr_loss = network.calc_loss(x, y)
        print("Epoch: %d  Loss: %.3f" % (epoch, curr_loss))

        if cv:
            val_acc = calc_accuracy(network, v_x, v_y)
            print("Validation Accuracy: %.3f\n" % val_acc)

        # perform backpropagation and weight update
        network.backward()

        # compute loss delta
        loss_delta = np.abs(curr_loss - prev_loss)

        # set next state
        prev_loss = curr_loss
        epoch += 1

    # Completion messages:
    if epoch > max_epochs:
        print("\n\nModel could not reach convergence ... Please increase max_epochs")
    else:
        print("\n\nModel reached convergence ... Training Complete !")

    # Final test accuracy:
    if test_x is not None and test_y is not None:
        print("Obtained Test set accuracy: %.3f" % calc_accuracy(network, t_x, t_y))


def parse_arguments():
    """
    Command line argument parser function
    :return: args => parsed arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_samples", action="store", type=int, default=30000,
                        help="Number of samples of synthetic data to be generated")
    parser.add_argument("--val_samples", action="store", type=int, default=500,
                        help="Number of validation samples")
    parser.add_argument("--test_samples", action="store", type=int, default=3000,
                        help="Number of test samples")
    parser.add_argument("--input_dims", action="store", type=int, default=12,
                        help="Dimensionality of input space")
    parser.add_argument("--n_classes", action="store", type=int, default=3,
                        help="Number of classes for classification")
    parser.add_argument("--data_gen_bias", action="store", type=int, default=100,
                        help="How far apart data clusters should be")
    parser.add_argument("--learning_rate", action="store", type=float, default=0.1,
                        help="Learning rate for gradient descent")
    parser.add_argument("--depth", action="store", type=int, default=3,
                        help="Depth of the Network")
    parser.add_argument("--max_epochs", action="store", type=int, default=300,
                        help="Max number of epochs for gradient descent")
    parser.add_argument("--widths", action="store", default=[32, 32, 32], nargs="*",
                        help="List of network widths. Note: [len(widths) == depth]")
    parser.add_argument("--validate", action="store", type=bool, default=True,
                        help="Boolean for validation")
    parser.add_argument("--convergence_threshold", action="store", type=float, default=1e-4,
                        help="Convergence threshold for gradient descent")

    args = parser.parse_args()

    return args


def main(args):
    """
    Main function for the script
    :param args: parsed command line arguments
    :return: None
    """

    # generate some random data samples
    x, y = generate_random_data_sample(args.n_samples,
                                       args.input_dims, args.n_classes,
                                       bias=args.data_gen_bias)

    val_x, val_y = generate_random_data_sample(args.val_samples,
                                               args.input_dims, args.n_classes,
                                               bias=args.data_gen_bias)

    test_x, test_y = generate_random_data_sample(args.test_samples,
                                                 args.input_dims, args.n_classes,
                                                 bias=args.data_gen_bias)

    # convert widths to an integer list
    args.widths = list(map(int, args.widths))

    nn = NeuralNetwork(
            f_dim=args.input_dims,
            n_classes=args.n_classes,
            depth=args.depth,
            widths=args.widths,
            learning_rate=args.learning_rate,
            activation_fn=lambda k: k
         )

    # test train method
    train_network(nn, x, y, val_x, val_y,
                  test_x, test_y,
                  max_epochs=args.max_epochs,
                  conv_thresh=args.convergence_threshold,
                  cv=args.validate)


if __name__ == '__main__':
    main(parse_arguments())
