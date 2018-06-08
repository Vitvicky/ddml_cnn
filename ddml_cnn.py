import logging
from itertools import combinations_with_replacement
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import tensor
from torch.utils.data import Dataset, DataLoader
# import torchvision
# from torchvision import transforms


class DDMLDataset(Dataset):
    """
    Implement a Dataset.
    """

    def __init__(self, dataset):
        """

        :param dataset: numpy.ndarray
        """

        self.data = []

        for s in dataset:
            x = (tensor(s[:-1], dtype=torch.float) / 255).view(-1, 28, 28)
            y = tensor(s[-1], dtype=torch.long)
            self.data.append((x, y))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class DDMLNet(nn.Module):
    def __init__(self, device, beta=1.0, tao=5.0, b=1.0, learning_rate=0.001):
        super(DDMLNet, self).__init__()
        self.conv1 = nn.Sequential(
            # [batch_size, 1, 28, 28]
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
            nn.ReLU(),
            # [batch_size, 6, 28, 28]
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            # [batch_size, 6, 14, 14]
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            # [batch_size, 16, 14, 14]
            nn.MaxPool2d(2, 2)
        )
        # [batch_size, 16, 7, 7]
        self.fc1 = nn.Linear(16 * 7 * 7, 1568)
        self.fc2 = nn.Linear(1568, 392)
        self.fc3 = nn.Linear(392, 10)

        self.ddml_layers = [self.fc1, self.fc2]

        self._s = F.tanh

        self.device = device

        self.beta = beta
        self.tao = tao
        self.b = b
        self.learning_rate = learning_rate

        self.logger = logging.getLogger(__name__)

        self.to(device)

    def cnn_forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 16 * 7 * 7)
        return x

    def ddml_forward(self, x):
        x = self.cnn_forward(x)
        x = self._s(self.fc1(x))
        x = self._s(self.fc2(x))
        return x

    def forward(self, x):
        x = self.ddml_forward(x)
        x = self.fc3(x)
        return x

    def compute_distance(self, x1, x2):
        """
        Compute the distance of two samples.
        ------------------------------------
        :param x1: Tensor
        :param x2: Tensor
        :return: The distance of the two sample.
        """
        return (self.ddml_forward(x1) - self.ddml_forward(x2)).data.norm() ** 2

    def ddml_optimize(self, pairs):
        """

        :param pairs:
        :return: loss.
        """
        loss = 0.0
        layer_count = len(self.ddml_layers)

        params_W = []
        params_b = []

        for layer in self.ddml_layers:
            params = list(layer.parameters())

            params_W.append(params[0])
            params_b.append(params[1])

        # calculate z(m) and h(m)
        # z(m) is the output of m-th layer without function tanh(x)
        # h(m) start from 0, which is m-1
        z_i_m = [[0 for m in range(layer_count)] for index in range(len(pairs))]
        h_i_m = [[0 for m in range(layer_count + 1)] for index in range(len(pairs))]
        z_j_m = [[0 for m in range(layer_count)] for index in range(len(pairs))]
        h_j_m = [[0 for m in range(layer_count + 1)] for index in range(len(pairs))]

        for index, (si, sj) in enumerate(pairs):
            xi = self.cnn_forward(si[0].unsqueeze(0))
            xj = self.cnn_forward(sj[0].unsqueeze(0))
            h_i_m[index][-1] = xi
            h_j_m[index][-1] = xj
            for m, layer in enumerate(self.ddml_layers):
                xi = layer(xi)
                xj = layer(xj)
                z_i_m[index][m] = xi
                z_j_m[index][m] = xj
                xi = self._s(xi)
                xj = self._s(xj)
                h_i_m[index][m] = xi
                h_j_m[index][m] = xj

        # calculate delta_ij(m)
        # calculate delta_ji(m)
        delta_ij_m = [[0 for m in range(layer_count)] for index in range(len(pairs))]
        delta_ji_m = [[0 for m in range(layer_count)] for index in range(len(pairs))]

        # M = layer_count, then we also need to project 1,2,3 to 0,1,2
        M = layer_count - 1

        # calculate delta(M)
        for index, (si, sj) in enumerate(pairs):
            xi = si[0].unsqueeze(0)
            xj = sj[0].unsqueeze(0)
            yi = si[1]
            yj = sj[1]

            # calculate c and loss
            if int(yi) == int(yj):
                l = 1
            else:
                l = -1

            dist = self.compute_distance(xi, xj)
            c = self.b - l * (self.tao - dist)
            loss += self._g(c)

            # h(m) have M + 1 values and m start from 0, in fact, delta_ij_m have M value and m start from 1
            delta_ij_m[index][M] = (self._g_derivative(c) * l * (h_i_m[index][M] - h_j_m[index][M])) * self._s_derivative(z_i_m[index][M])
            delta_ji_m[index][M] = (self._g_derivative(c) * l * (h_j_m[index][M] - h_i_m[index][M])) * self._s_derivative(z_j_m[index][M])

        loss /= len(pairs)

        # calculate delta(m)
        for index in range(len(pairs)):
            for m in reversed(range(M)):
                delta_ij_m[index][m] = torch.mm(delta_ij_m[index][m + 1], params_W[m + 1]) * self._s_derivative(z_i_m[index][m])
                delta_ji_m[index][m] = torch.mm(delta_ji_m[index][m + 1], params_W[m + 1]) * self._s_derivative(z_j_m[index][m])

        # calculate partial derivative of W
        partial_derivative_W_m = [0 for m in range(layer_count)]

        for m in range(layer_count):
            for index in range(len(pairs)):
                partial_derivative_W_m[m] += torch.mm(delta_ij_m[index][m].t(), h_i_m[index][m - 1])
                partial_derivative_W_m[m] += torch.mm(delta_ji_m[index][m].t(), h_j_m[index][m - 1])

        # calculate partial derivative of b
        partial_derivative_b_m = [0 for m in range(layer_count)]

        for m in range(layer_count):
            for index in range(len(pairs)):
                partial_derivative_b_m[m] += (delta_ij_m[index][m] + delta_ji_m[index][m]).squeeze()

        for m, layer in enumerate(self.ddml_layers):
            params = list(layer.parameters())
            params[0].data.sub_(self.learning_rate * partial_derivative_W_m[m])
            params[1].data.sub_(self.learning_rate * partial_derivative_b_m[m])

        return loss

    def _g(self, z):
        """
        Generalized logistic loss function.
        -----------------------------------
        :param z:
        """
        z = torch.tensor(z)
        if z > 10:
            value = z
        else:
            value = torch.log(1 + torch.exp(self.beta * z)) / self.beta
        return value

    def _g_derivative(self, z):
        """
        The derivative of g(z).
        -----------------------
        :param z:
        """
        z = torch.tensor(z)
        return 1 / (torch.exp(-1 * self.beta * z) + 1)

    def _s_derivative(self, z):
        """
        The derivative of tanh(z).
        --------------------------
        :param z:
        """
        z = torch.tensor(z)
        return 1 - self._s(z) ** 2


def setup_logger(level=logging.DEBUG):
    """
    Setup logger.
    -------------
    :param level:
    :return: logger
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def train(net, dataloader, epoch_number=2):
    logger = logging.getLogger(__name__)

    average_number = 100

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epoch_number):
        cnn_loss = 0.0
        ddml_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(net.device), labels.to(net.device)
            pairs = list(combinations_with_replacement(zip(inputs, labels), 2))

            ################
            # cnn backward #
            ################
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            #################
            # ddml backward #
            #################
            ddml_loss += net.ddml_optimize(pairs)

            # print statistics
            cnn_loss += loss.item()
            if (i + 1) % average_number == 0:
                logger.debug('(%d, %5d) cnn loss: %.3f, ddml loss: %.3f', epoch, i + 1, cnn_loss / average_number, ddml_loss / average_number)
                cnn_loss = 0.0
                ddml_loss = 0.0

    logger.info('Finished Training')


def test(net, dataloader):
    currcet = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(net.device), labels.to(net.device)
        outputs = net(inputs)
        value, result = torch.max(outputs, dim=1)

        if result == labels:
            currcet += 1

    return currcet / len(dataloader)


if __name__ == '__main__':
    LOGGER = setup_logger(level=logging.DEBUG)

    train_batch_size = 10

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # DEVICE = torch.device("cpu")

    # transform = transforms.Compose([transforms.ToTensor()])
    #
    # trainset = torchvision.datasets.FashionMNIST(root="data/fashion-mnist", train=True, transform=transform, download=False)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    # testset = torchvision.datasets.FashionMNIST(root="data/fashion-mnist", train=False, transform=transform, download=False)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=1, num_workers=2)

    dataset = np.loadtxt('data/fashion-mnist_train.csv', delimiter=',')
    LOGGER.debug("Dataset shape: %s", dataset.shape)

    trainset = DDMLDataset(dataset[:5000])
    testset = DDMLDataset(dataset[5000:])

    trainloader = DataLoader(dataset=trainset, batch_size=train_batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(dataset=testset, batch_size=1, shuffle=False, num_workers=0)

    cnnnet = DDMLNet(device=DEVICE, beta=1.0, tao=20.0, b=2.0, learning_rate=0.001)

    train(cnnnet, trainloader, epoch_number=4)
    accuracy = test(cnnnet, testloader)

    LOGGER.info("Accuracy: %6f", accuracy)
