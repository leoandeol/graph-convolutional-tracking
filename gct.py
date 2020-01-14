
import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import alexnet

from layers import GraphConvolution
from functions import regularized_laplacian

from math import sqrt


class GCT(nn.Module):

    def __init__(self, T):
        super(GCT, self).__init__()
        #### Constants
        # Number of parts of exemplar : D1 x 1 x 1
        self.M_z = 36 # 6**2
        # Number of parts of search image
        self.M_x = 484 # 22**2
        #Feature dimensionality of embeddings
        self.D_1 = 256
        self.D_2 = 256
        # Time range
        self.T = T
        
        #temporary measure until automatically resized input pics
        self.M_x = 88
        self.M_z = 88
    
        # Shared convnet "phi" and block 3 first conv layers
        self.shared_convnet = alexnet(pretrained=True).features
        for param in self.shared_convnet[:8]:
            param.requires_grad=False
        # Probably unnecessary : self.shared_convnet.add_module("13",nn.Conv2d(256,self.D_1,3)) 
        
        # Instance Context featuring : gotta check that
        self.conv_deconv = nn.Sequential(nn.Conv1d(self.D_1, self.D_2, 3, padding=1),
                                         nn.MaxPool1d(self.M_x),
                                         nn.ConvTranspose1d(self.D_1, self.D_2, self.M_z))
        
        # ST-GCN 
        self.st_gcn = GCN(self.D_1,512,self.D_2)
        self.st_pool = nn.MaxPool1d(self.T)
        
        # Graph Learning : 3 or 1 input channel ?
        self.g = nn.Conv1d(self.D_1,self.D_1,1)
        self.h = nn.Conv1d(self.D_1,self.D_1,1)
        
        # CT-GCN
        self.ct_gcn = GCN(self.D_2,384,self.D_2)
        
        
        
    def __call__(self, search, exemplars):
        return self.forward(search,exemplars)

        
    def forward(self, search, exemplars):
        #TODO : consider T
        # Embeddings
        X, Z = self.shared_convnet(search[None,:,:,:]), self.shared_convnet(exemplars)
        print("X",X.shape,"Z",Z.shape)
        # Part Division 
        Z = Z.view(Z.shape[0], Z.shape[1], -1)
        X = X.view(1, X.shape[1], -1)
        print("X",X.shape,"Z",Z.shape)
        # Context feature of search image

        
        #tmp = nn.Conv1d(self.D_1, self.D_2, 3, padding=1)(X)
        #tmp = nn.MaxPool1d(self.M_x)(tmp)
        #X_hat = nn.ConvTranspose1d(self.D_1, self.D_2, self.M_z)(tmp)
                           
        X_hat = self.conv_deconv(X)
        print("X^",X_hat.shape)
        # ST-GCN
        A_1 = self.build_st_graph(self.T, self.M_z)
        V_1 = self.st_gcn(Z.permute(0,2,1), A_1).permute(0,2,1)
        V_1 = self.st_pool(V_1.permute(2,1,0)).permute(2,1,0) #time range T max pooling
        
        print("A_1",A_1.shape,"\nV_1",V_1.shape)
        # Graph Learning 
        V_x = V_1 + X_hat
        print("V_x",V_x.shape)
        A_2 = self.build_ct_graph(V_x)
        print("V_x",V_x.shape,"A_2",A_2.shape)
        # CT-GCN
        V_2 = self.ct_gcn(V_1.permute(0,2,1), A_2).permute(0,2,1)

        print("V_2", V_2.shape)
        # XCorr
        a, b, c = X.shape[0], X.shape[1], int(sqrt(X.shape[2]))
        R = F.conv2d(X.view((a,b,11,8)),V_2.reshape((a,b,11,8)))
        
        return R


    def build_st_graph(self, T, M_z):
        # Make sparse 
        A = torch.zeros((T*M_z,T*M_z))#, layout=torch.sparse_coo)
        for i in range(T):
            for j in range(M_z):
                for k in range(M_z):
                    if j != k:
                        A[i + T*j, i + T*k]=1
                        A[i + T*k, i + T*j]=1
        for j in range(M_z):
            for i in range(T-1):
                A[i+1 + T*j, i+1 + T*j]=1
        return A
        
        
    def build_ct_graph(self, V_x):
        # make it sparse
        d = V_x.shape[-1]
        A = torch.zeros((d,d))#, layout=torch.sparse_coo)
        #print(V_x.shape)
        #print(self.g(V_x[:,:,0][:,:,None]).shape)
        for i in range(self.M_z):
            for j in range(self.M_z):
                A[j,i]=torch.mm(self.g(V_x[:,:,i,None])[0,:,:].T,self.h(V_x[:,:,j,None])[0,:,:])
        #normalize
        A -= A.max(1)[0][:,None]
        A = torch.exp(A)
        A = A / A.sum(1)[:,None]
        return A
    
class GCN(nn.Module):

    def __init__(self, d_in, d_hid, d_out):
        super(GCN, self).__init__()
        self.conv1 = GraphConvolution(d_in,d_hid)
        self.act1 = nn.LeakyReLU()
        self.conv2 = GraphConvolution(d_hid,d_out)
        self.act2 = nn.LeakyReLU()

    def forward(self, x, adj):
        x = self.act1(self.conv1(x,adj))
        x = self.act2(self.conv2(x,adj))
        return x

    
