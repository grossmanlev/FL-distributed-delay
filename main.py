import copy
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from torch.utils.tensorboard import SummaryWriter

from FL.agents import *
from FL.models import *
from FL.util import *


# Aggregator rule: mean
def rule(ups_list):  # ups_list is a list of list of tensors
    return [torch.stack([x[i] for x in ups_list]).mean(0)
            for i in range(len(ups_list[0]))]


def noniid_batch_trainset(trainset, targets):
    """ Return trainset whose targets are in list c """
    indices = np.where(np.isin(np.array(trainset.targets), targets))[0]
    trainset2 = copy.deepcopy(trainset)
    trainset2.data = trainset2.data[indices]
    trainset2.targets = np.array(trainset2.targets)[indices]
    return trainset2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_workers', type=int, default=5,
                        help='number of FL workers')
    parser.add_argument('--n_epochs', type=int, default=1000,
                        help='number of train epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='number of train epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='number of train epochs')
    parser.add_argument('--delay', type=int, default=100,
                        help='delay in between slow worker')
    parser.add_argument('--model_dir', type=str,
                        help='path to model to load (pretrained)')
    parser.add_argument('--throttle', action='store_true',
                        help='toggle whether to use gradient throttling')
    args = parser.parse_args()

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()

    n_workers = args.n_workers
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    noniid = True
    load_model = True
    save_model = False

    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Import Datasets
    trainset = torchvision.datasets.CIFAR10(
        root='../data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root='../data', train=False, download=True, transform=transform)

    # slow_guys = 9
    # Batch Loaders
    if noniid:
        targets = [[0, 2, 3, 4, 5, 6, 7, 8], [1, 9]]
        trainsets = [noniid_batch_trainset(trainset, targets[0]) for _ in range(n_workers - 1)]
        trainsets.append(noniid_batch_trainset(trainset, targets[1]))
        # trainsets = [noniid_batch_trainset(trainset, i) for i in set(trainset.targets)]
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
        model.load_state_dict(torch.load("saved_model_100.pt"))
        model.eval()

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

    pesky_worker_grads = []

    throttle_window = 0
    max_throttle = 32
    slow_guy_gone = False

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

            if False:
                worker_list[i].model = central.model
                ups, loss = worker_list[i].fwd_bkwd(batch_inp, batch_outp)
                weight_ups.append(ups)
                losses.append(loss)
            else:
                if i == n_workers - 1:
                    slow_guy_gone = False
                    ups = None
                    if t == 0:
                        worker_list[i].model = central.model
                        ups, loss = worker_list[i].fwd_bkwd(batch_inp, batch_outp)
                        pesky_worker_grads.append(ups)
                        ups = None
                    elif t % args.delay == 0:
                        worker_list[i].model = central.model
                        ups, loss = worker_list[i].fwd_bkwd(batch_inp, batch_outp)
                        pesky_worker_grads.append(ups)
                        ups = pesky_worker_grads.pop(0)

                    if ups is not None:
                        weight_ups.append(ups)
                        slow_guy_gone = True
                else:
                    if throttle_window <= 0:
                        worker_list[i].model = central.model
                        ups, loss = worker_list[i].fwd_bkwd(batch_inp, batch_outp)
                        weight_ups.append(ups)
                        losses.append(loss)

                        if args.throttle:
                            throttle_window = 1
                            if not slow_guy_gone:
                                throttle_window *= 2
                                throttle_window = min(throttle_window, max_throttle)

            if throttle_window > 0:
                throttle_window -= 1

        # Aggregate Worker Gradients
        weight_ups_FIN = agg.rule(weight_ups)
        writer.add_scalar('Avg. Loss', np.mean(losses), t)

        # Update Central Model
        central.update_model(weight_ups_FIN)

        central.model.eval()

        if t % 100 == 0 and t > 0 and save_model:
            print('Saving model...')
            torch.save(central.model.state_dict(), "saved_model_{}.pt".format(t))

        if t % 100 == 0 and t > 0:
            all_accuracies = print_test_accuracy(model, testloader)
            avg_accuracy = np.mean(all_accuracies)
            epochs.append(t)
            accuracies.append(avg_accuracy)

            writer.add_scalar('Avg. Test Accuracy', avg_accuracy, t)
            writer.add_scalar('Class 9 Test Accuracy', all_accuracies[-1], t)

    all_accuracies = print_test_accuracy(model, testloader)
    avg_accuracy = np.mean(all_accuracies)
    accuracies.append(avg_accuracy)

    writer.add_scalar('Avg. Test Accuracy', avg_accuracy, t)
    writer.add_scalar('Class 9 Test Accuracy', all_accuracies[-1], t)

    print('Done training')
