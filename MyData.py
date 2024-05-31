import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder

# sMRInpyDataset class
class sMRInpyDataset(Dataset):
    def __init__(self, sMRIFilePath:str, transform=None):
        self.sMRIList = pd.read_csv(sMRIFilePath)
        self.sMRInpyList = self.sMRIList['npy_path'].to_list()
        self.APOE_A1 = self.sMRIList['APOE A1'].to_list()   # 2,3,4
        self.APOE_A2 = self.sMRIList['APOE A2'].to_list()   # 3,4
        self.labels = self.sMRIList['Research Group'].to_list()
        self.labels = LabelEncoder().fit_transform(self.labels)

    def __len__(self):
        return len(self.sMRInpyList)

    def __getitem__(self, idx):
        npyData = np.load('data/data/' + self.sMRInpyList[idx])
        npyData = self.transform(npyData)
        apoe = np.array([self.APOE_A1[idx], self.APOE_A2[idx]]) # [2 2], [2 3], [2 4], [3 3], [3 4], [4 4]

        if np.array_equal(apoe, [2, 2]):
            apoeL = 0
        elif np.array_equal(apoe, [2, 3]):
            apoeL = 1
        elif np.array_equal(apoe, [2, 4]):
            apoeL = 2
        elif np.array_equal(apoe, [3, 3]):
            apoeL = 3
        elif np.array_equal(apoe, [3, 4]):
            apoeL = 4
        elif np.array_equal(apoe, [4, 4]):
            apoeL = 5
        apoeLE = LabelEncoder().fit([0, 1, 2, 3, 4, 5])
        apoeLabel = apoeLE.transform([apoeL])
        # apoe onehot
        apoeLabel = torch.nn.functional.one_hot(torch.tensor(apoeLabel), num_classes=6).float()

        return (npyData, apoeLabel), self.labels[idx]
    
    def transform(self, sample):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            #transforms.Lambda(lambda x: x.float())
        ])
        return transform(sample)
    
if __name__ == '__main__':
    sMRIFilePath = 'idaSearch_with_npy_path.csv'
    sMRInpyDataset = sMRInpyDataset(sMRIFilePath)
    print('sMRInpyDataset:', sMRInpyDataset)
    print('sMRInpyDataset[0][0]:', sMRInpyDataset[0][0])
    print('sMRInpyDataset[0][1]:', sMRInpyDataset[0][1])
    print(sMRInpyDataset[0][0][1])