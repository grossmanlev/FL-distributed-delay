import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
import torch.nn.functional as F
import torchvision
import numpy as np
import time
from tqdm import tqdm

from FL.agents import *
from FL.models import *
from FL.util import *


# Aggregator rule: mean
def rule(ups_list):  # ups_list is a list of list of tensors
    return [torch.stack([x[i] for x in ups_list]).mean(0)
            for i in range(len(ups_list[0]))]


if __name__ == '__main__':
    n_workers = 5
    n_epochs = 1000
    batch_size = 128
    learning_rate = 0.001
    noniid = False
    load_model = False

    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Import Datasets
    trainset = torchvision.datasets.CIFAR10(
        root='../data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root='../data', train=False, download=True, transform=transform)

    # Batch Loaders
    if noniid:
        def noniid_batch_trainset(trainset, c):
            indices = (np.array(trainset.targets) == c)
            trainset2 = copy.deepcopy(trainset)
            trainset2.data = trainset2.data[indices]
            trainset2.targets = [c for i in range(len(indices))]

            return trainset2

        trainsets = [noniid_batch_trainset(trainset,i) for i in set(trainset.targets)]
    else:
        trainsets = [trainset]

    samplers = [torch.utils.data.RandomSampler(i, replacement=True) for i in trainsets]
    trainloaders = [torch.utils.data.DataLoader(
        trainsets[i], batch_size=batch_size, shuffle=False, sampler=samplers[i],
        num_workers=0) for i in range(len(trainsets))]

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Setup Learning Model
    model = PerformantNet1()
    if load_model:
        model.load_state_dict(torch.load("PerformantNet1_10epochs.pt"))
        n_epochs = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")
    print(device)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss = nn.CrossEntropyLoss()

    # Setup Federated Learning Framework
    central = Central(model, optimizer)
    worker_list = []
    for i in range(n_workers):
        worker_list.append(Worker(loss))
    agg = Agg(rule)

    epochs = []
    accuracies = []

    # Training Loop
    for t in tqdm(range(n_epochs)):
        first_time = time.time()

        weight_ups = []
        losses = []
        central.model.train()

        dataiters = [iter(trainloader) for trainloader in trainloaders]

        # Worker Loop
        for i in range(n_workers):
            k = np.random.randint(0, len(dataiters))
            dataiter = dataiters[k]

            batch_inp, batch_outp = dataiter.next()
            batch_inp, batch_outp = batch_inp.to(device), batch_outp.to(device)

            worker_list[i].model = central.model

            ups, loss = worker_list[i].fwd_bkwd(batch_inp, batch_outp)
            weight_ups.append(ups)
            losses.append(loss)

        # Aggregate Worker Gradients
        weight_ups_FIN = agg.rule(weight_ups)

        print('Avg. Loss: {}'.format(np.mean(losses)))

        # Update Central Model
        central.update_model(weight_ups_FIN)

        central.model.eval()

        if t > 0 and t % 100 == 0:
            print('Epoch: {}, Time to complete: {}'.format(t, time.time() - first_time))

        if t % 250 == 0 and t > 0:
            # print('Epoch: {}'.format(t))
            accuracy = print_test_accuracy(model, testloader)
            epochs.append(t)
            accuracies.append(accuracy)

    print('Done training')
