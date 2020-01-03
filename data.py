import os
import numpy as np
import matplotlib.pyplot as plt # or OpenCV for plot video
from skimage import io
import torch
from torch.utils.data import Dataset, DataLoader


class VOT(Dataset):

    def __init__(self,year):
        self.path = "data/vot"+str(year)+"/"
        
        with open(self.path+"list.txt","r") as f:
            self.sample_names = {x:y for x,y in enumerate(f.read().splitlines())}
            #CHARGER labels et tout

    def __len__(self):
        pass

    #todo : normalize data

    def __getitem__(self, idx):
        tensors = []
        name = self.sample_names[idx]
        for img in sorted(os.listdir(self.path+name+"/color/")):
            image = io.imread(self.path+name+"/color/"+img)
            tensors.append(torch.tensor(image.astype(np.float32)/255,dtype=torch.float).T)
        with open(self.path+name+"/groundtruth.txt","r") as f:
            boxes = np.zeros((len(tensors),8))
            for i,line in enumerate(f.read().splitlines()):
                boxes[i] = np.array(line.split(","))
        return torch.stack(tensors), boxes


class VOT17(VOT):
    def __init__(self):
        super(VOT17, self).__init__(17)

class VOT19(VOT):
    def __init__(self):
        super(VOT19, self).__init__(19)

class ILSVRC2015Video(Dataset):

    def __init__(self):
        pass
    # 80 GB
    #todo : http://bvisionweb1.cs.unc.edu/ilsvrc2015/download-videos-3j16.php#vid
        
if __name__=="__main__":
    print("Testing loading of datasets")
    v7 = VOT17()
    v9 = VOT19()


    a,b = v9.__getitem__(1)
    print(a.shape,b.shape)
