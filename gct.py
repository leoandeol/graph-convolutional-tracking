import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import alexnet

class GCT(nn.Module):

    def __init__(self):
        self.D_1 = 256
        self.D_2 = ...
        #Spatial_temporal graph ???
    
        # Shared convnet "phi"
        self.share_convnet = alexnet(pretrained=True)
        #todo bloquer les deux premieres couches et fine tune les derux dernieres, et ajouter 3x3 to descendre channel a 256 D_1
        
        # Instance Context featuring
        self.conv_deconv = nn.Sequential(..)
        
        # Part divisoion = ???
        
        # ST-GCN 
        self.st_gcn = ... 
        
        # Graph Learning : 3 or 1 input channel ?
        self.g = nn.Conv2D(3,D_1,1)
        self.h = nn.Conv2D(3,D_1,1)
        
        # CT-GCN
        self.ct_gcn = ...
        
        
        
    def __call__(self, search, exemplars):
        return self.forward(search,exemplars)

        
    def forward(self, search, exemplars):
        # Embeddings
        X, Z = self.shared_convnet(search), self.shared_convnet(exemplars)
        
        # Context feature of search image
        X_hat = self.conv_deconv(X)
        
        # Part Division 
        Z = ...#TODO
        
        # ST-GCN
        V_1 = self.st_gcn(Z, A_1)
        
        # Graph Learning 
        Vx = V + X_hat
        A_2 = self.graph_learning(Vx)       
        
        # CT-GCN
        V_2 = self.ct_gcn(V_1, A_2)
        
        # XCorr
        R = ...
        
        return R
        
        
    def graph_learning(self, V_x):
        # check dimensions and if torch does elementwise exponent
        d = V_x.shape[-1]
        A_2 = torch.zeros((d,d))
        g, h = self.g(
    