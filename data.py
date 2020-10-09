"""
How to do data?
===============
The entire weather dataset cannot be loaded completely in memory
so this has to be either
A. iterative datatset where we yield samples
B. we can store samples in a seperate file and load when required
C. read a particular line from file lazily (using linecache)
D. use a dedicated system like hdf5

Option D sounds the simples as it is already is used to save the scientific
data like earth observation, etc.

Going by option D requires these tasks
#1 learning what the hell is HDF5
#2 creating a script for unified datasets and

Part #1
-------
So HDF5 is short for heirarchical data format (v)5. HDF5 uses a directory
style method to store datasets and 

/ # root
/headers # first group


Part #2
-------
Next step is simple, need to crate two files each 


# 15.09.2020 - @yashbonde
"""

import re
import json
import numpy as np
import pandas as pd
import sentencepiece as spm

import torch
from torch.utils.data import Dataset

# ---- functions ---- #

# ---- main class ---- #
class WeatherModelingDataset(Dataset):
    def __init__(self, config, mode = "train"):
        self.config = config

        print("Loading samples ...")
        with open(config.path, "r") as f:
            data = json.load(f)

        self.data = [data[i:i+config.maxlen] for i in range(len(data) - config.maxlen)]
        print(f"Dataset [{mode}] {len(self.data)} samples")

    def __len__(self):
        return len(self.data)
    
  
    def __getitem__(self, index):
        """
        It is getting too complicated to filter and create datasets, so I have started
        with directly creating the dataset object. We need to return the following objects
        with these shapes (T - timesteps, N - number of nodes):
            # this is the real challenge
            input: [T, N, 16]
            locations: [T, N, 3]
            node_mask: [T, N, N] # this tells what are the nodes that are active at each point of time.

            # these values are like the standard stuff from LM
            month_ids: [T,]
            day_ids: [T,]
            hour_ids: [T,]

            # this is constant
            edge_matrix: [N, N]
        """
        # get samples

        # parsing 

        # generate the tensors

        # return a dict object

        return {
            "input": [], # [T, N, F]
            "locations", # [T, N, F]

            "node_mask": # [T, N, N]

            "edge_matrix": , # [N, N]

            "month_ids", # [T,]
            "day_ids", # [T,]
            "hour_ids", # [T,]
            
        }
       

class DatasetConfig:
    path = None
    sheets = None
    size = 3000 # since we generate data runtime size doesn't matter

    # about probabilities of the fields
    pf = 0.5 # probability of adding fields in sample
    fmax = 0.8 # at max include only 80 of fields
    fmin = 0.1 # atleast include 10 of fields
    
    proc = "sample"

    maxlen = None # what is the maximum length of sequence to return

    def __init__(self, **kwargs):
        self.attrs = []
        for k,v in kwargs.items():
            setattr(self, k, v)
            self.attrs.append(k)

    def __repr__(self):
        return "---- DATASET CONFIGURATION ----\n" + \
            "\n".join([f"{k}\t{getattr(self, k)}" for k in list(set([
                "path",
                "sheets",
                "size",
                "pf",
                "fmax",
                "fmin",
                "maxlen",
                "proc"
            ] + self.attrs)) 
        ]) + "\n"
