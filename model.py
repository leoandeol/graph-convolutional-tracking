import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GCTModel(nn.Module):

    def __init__(self):
        self.shared_convnet = nn.Sequential(nn.Conv2d(),
                                           nn.Batchnorm2d(),
                                           nn.Relu(),
                                           nn.MaxPool2d(),
                                           nn.Conv2d(),
                                           nn.Batchnorm2d(),
                                           nn.Relu(),
                                           nn.MaxPool2d(),
                                           nn.Conv2d(),
                                           nn.Batchnorm2d(),
                                           nn.Relu(),
                                           nn.MaxPool2d())

        self.current_conv = nn.Sequential(nn.Conv2d(),
                                           nn.Batchnorm2d(),
                                           nn.Relu(),
                                           nn.MaxPool2d())
        self.current_deconv = nn.Sequential(ConvTranspose2d())
        

    def forward(self, current_image, historical_images):
        #Computation on current search image
        instance_embedding = self.shared_convnet(current_image)
        context_feature = self.current_deconv(self.current_conv(instance_embedding))
        #Computation on Historical Examplar
        examplars_embeddings = self.share_convnet(historical_images)
        examplars_divided = #TODO
        st_feature = #TODO self.st_gcn(examplars_dvidid, st_graph)

        #Merge on Context
        graph = context_feature + st_feature
        #TODO GRAPH LEARNING FOR ADAPTIVE GRAPH
        adaptive_feature = #TODO self.ct_gcn(st_feature,adaptive_graph)

        #Compute the response
        XCorr = #TODO
        response_map = #TODO

        return response_map
        

                                    
