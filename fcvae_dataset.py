# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 10:31:01 2020

@author: hiltunh3
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.autograd import Variable

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import seaborn as sns
import random
import glob

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(filepath, test_size: float = 0.1, batch_size: int = 64, validation: bool = False):
    #files = glob.glob(filepath + "train_" + "*.txt")
    files = glob.glob(filepath + "*.txt")
    all_datasets = []
    for i, file in enumerate(files): #order of markers, missing cloumns
        fc = pd.read_csv(file, delimiter = ',')
        fc.iloc[:,[1,2,3,4,5,6]] = np.log(fc.iloc[:,[1,2,3,4,5,6]]) 
        std_dev = np.std(fc)
        mean = np.mean(fc)
        fc = (fc - mean) / std_dev
        fc_sub = fc.iloc[:,[0,1,4, 2,3,5,6]]
        
        #ratio_remaining = 1 - test_size
        #ratio_val = test_size / ratio_remaining

        #train, test = train_test_split(fc_sub, test_size=ratio_val)
        train, test = train_test_split(fc_sub, test_size=test_size) 
        train_data = fcm_data(train)
        test_data = fcm_data(test)
        all_data = fcm_data(fc_sub)
        train_iterator = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_iterator = DataLoader(test_data, batch_size=batch_size)
        whole_dataset = DataLoader(all_data, batch_size=batch_size)
        all_datasets.append([fc_sub, train_data, test_data, train_iterator, test_iterator, whole_dataset])   

    return all_datasets

class fcm_data(Dataset):
    
    def __init__(self, data):
        self.data = data
        self.marker_names = np.asarray(data.columns, dtype=str)
        self.cell_names = np.asarray(data.index.values, dtype=str)
        self.data = data.values
        
    def __len__(self):
        return len(self.cell_names)
    
    def __getitem__(self, index):
        return self.data[index,:]
    
    @property 
    def nb_cells(self) -> int:
        return self.data.shape[0]

    @property
    def nb_markers(self) -> int:
        return self.data.shape[1]
