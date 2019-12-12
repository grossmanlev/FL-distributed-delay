# Distributed Delay
## CS143 Final Project

### Overview

This repository contains all the code written for the CS143 final project. All saved experiments can be found within our ```runs_<experiment type>``` folders, and all experiments can be rerun by following the instructions below.

### Installation and Prerequisites

First, a number of prerequisites must be installed:

```
python3
numpy
tqdm
torch
torchvision
tensorboard
```

After installation, you should be all set.

### Testing

The main file by which all experiments can be run is ```main.py```. This file takes in a number of optional command-like arguments, all of which can be viewed by running

```
python3 main.py -h
```

Following are a few example tests one can run:

1) Run for 1500 epochs using 10 workers, one of which has an artificial delay of 50
```
python3 main.py --n_epochs 1500 --n_workers 10 --delay 50
```

2) Run using the same setup, except start with a PyTorch pre-trained/warm-started central model found in `warm_start.pt`

```
python3 main.py --n_epochs 1500 --n_workers 10 --delay 50 --model_file warm_start.pt
```

3) Run using the same setup, except now with gradient throttling enabled
```
python3 main.py --n_epochs 1500 --n_workers 10 --delay 50 --throttle
```

NOTE: All training losses and accuracies can be checked in real-time using tensorboard:
```
tensorboard --logdir=runs
```

Plus, all previously run experiments can be viewed by running, for example:
```
tensorboard --logdir=runs_delay
```
