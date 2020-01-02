import numpy as np
import torch 
import torchvision.datasets
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(prog="GCT",description="Graph Convolutional Tracking")
#Revoir ces params
parser.add_argument("--dataset", default="mnist", type=str, help="Name of the target dataset")
parser.add_argument("--n_epochs", default=60, type=int, help="Number of epochs")
parser.add_argument("--batch_size", default=128, type=int, help="Size of batch")
parser.add_argument("--learning_rate", default=0.0002, type=float)
parser.add_argument("--seed", default=42, type=int)

parser.add_argument('--cuda', dest='cuda', action='store_true')
parser.add_argument('--no-cuda', dest='cuda', action='store_false')
parser.set_defaults(cuda=torch.cuda.is_available())
parser.add_argument('--verbose', dest='verbose', action='store_true')
parser.set_defaults(verbose=False)

args = parser.parse_args()

torch.random.manual_seed(args.seed)
np.random.seed(args.seed)
