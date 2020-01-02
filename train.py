import numpy as np
import torch 
import torchvision.datasets
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(prog="GCT",description="Graph Convolutional Tracking")
parser.add_argument("--n_epochs", default=50, type=int, help="Number of epochs")
parser.add_argument("--batch_size", default=24, type=int, help="Size of batch")
parser.add_argument("--learning_rate", default=0.005, type=float)
parser.add_argument("--weight_decay", default=5e-5, type=float)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument('--cuda', dest='cuda', action='store_true')
parser.add_argument('--no-cuda', dest='cuda', action='store_false')
parser.set_defaults(cuda=torch.cuda.is_available())
parser.add_argument('--verbose', dest='verbose', action='store_true')
parser.set_defaults(verbose=False)
parser.add_argument('--download', dest='download', action='store_true')
parser.set_defaults(download=False)

args = parser.parse_args()

torch.random.manual_seed(42)
np.random.seed(42)

# Offline train on ILSVRC2015 on about 2015 videos totalling over 1 million annotated frames.
# In each video we collect each training sample of T+1 frames within the nearest 100 frames.
# We use the former T frames as exemplar images and take the last o,ne as the search image.
# Adam with lr = 0.005 & weight decay = 5e-5
# 50 epochs, batch size of 24
  
