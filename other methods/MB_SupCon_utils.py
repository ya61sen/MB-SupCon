#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import os
import math
import numpy as np
import pandas as pd
import re
import sys

from scipy.stats import spearmanr
#import skimage

import time
import torch
import torch.hub
import torch.nn
import torch.utils
import torch.utils.data

class Dataset(torch.utils.data.Dataset):
    __initialized = False
    def __init__(self, indexes, data_dict):
        """
        Args:
            indexes: shared indexes across different datasets
        """
        self.indexes = indexes
        self.indexes_dict = dict()
        for i, _ in enumerate(indexes):
            self.indexes_dict[_] = i
        
        self.keys = data_dict.keys()
        self.data_dict = data_dict
        self.__initialized = True

    def __len__(self):
        """Denotes the number of samples"""
        return len(self.indexes)
    
    def __getitem__(self, index):
        """Generate one batch of data.
        
        Returns:
            idx: indexes of samples (long)
        """
        # Generate indexes of the batch
        data_index = self.indexes[index]
        
        # Generate torch.long indexes of the batch samples
        idx = self.indexes_dict[data_index]

        # Generate data
        data = self.__data_generation(data_index)

        return data, idx
    
    def __data_generation(self, indexes):
        """Generates data containing batch_size samples.
        
        Returns:
            data: data.g in [b, n_microbes]; data.m in [b, n_metabolites]
        """
        
        data = dict()
        for key in self.keys:
            data[key] = torch.tensor(self.data_dict[key].loc[indexes, :].values)
        
        return data


class Encoder(torch.nn.Module):
    def __init__(self, net_dict, n_out_features, dropout_rate):
        """
        Args:
            net_dict: e.g., {gut_16s: [10000, 100, 100]} to define the structures
                      the keys should exist in keys for DataLoader
            n_out_features: number of output features
        """
        super(Encoder, self).__init__()
        
        self.keys = net_dict.keys()
        
        for key in self.keys:
            net_fcs = []
            structure = net_dict[key]
            net_in_shape = structure[0]
            for i, net_n_hidden_nodes in enumerate(structure[1:]):
                net_fcs.append(torch.nn.Linear(net_in_shape, net_n_hidden_nodes))
                net_fcs.append(torch.nn.BatchNorm1d(net_n_hidden_nodes))
                net_fcs.append(torch.nn.ReLU6())
                net_fcs.append(torch.nn.Dropout(p=dropout_rate))
                net_in_shape = net_n_hidden_nodes
            net_fcs.append(torch.nn.Linear(net_in_shape, n_out_features))
            setattr(self, key, torch.nn.ModuleList(net_fcs))
        
    def forward(self, data):
        """
        Args:
            data: a dictionary
        """
        f = dict()
        
        for key in self.keys:
            _f = data[key]
            net_fcs = getattr(self, key)
            for net_fc in net_fcs:
                _f = net_fc(_f)
            f[key] = _f
        
        return f
