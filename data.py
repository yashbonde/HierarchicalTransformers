"""
How to do data?
===============
The entire weather dataset cannot be loaded completely in memory
so this has to be either
A. iterative datatset where we yield samples [no]
B. we can store samples in a seperate file and load when required [no]
C. read a particular line from file lazily (using linecache) [no]
D. use a dedicated system like hdf5 [too time consuming to build]

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
import h5py
import math
import numpy as np
import pandas as pd
import sentencepiece as spm

import torch
from torch.utils.data import Dataset

# ---- functions ---- #

_convert_to_radians = lambda angle: math.radians(float(angle))

def convert_latlong_to_great_circle(lat_long_1, lat_long_2):
    """convert the Latitude Longitude to Distance on earth"""
    (p1, l1), (p2, l2) = lat_long_1, lat_long_2
    p1, l1, p2, l2 = list(map(_convert_to_radians, (p1, l1, p2, l2)))
    del_l = l1 - l2
    central_angle = math.atan(math.sqrt(
        ((math.cos(p2)*math.sin(del_l)) ** 2) + ((math.cos(p1)*math.sin(p2) - math.sin(p1)*math.cos(p2)*math.cos(del_l)) ** 2)
    ) / (math.sin(p1)*math.sin(p2) + math.cos(p1)*math.cos(p2)*math.cos(del_l)))
    dist =  6371 * central_angle
    return dist

# ---- main class ---- #
class WeatherDataset(Dataset):
    def __init__(self, config, mode="train"):
        self.config = config
        self.hdf = h5py.File(config.hdf_fpath, mode="r")

        with open(config.index, "r") as f:
            self.this_idx = [int(l.strip()) for l in f.readlines()]

        self.set_location_and_edge_matrix()
        self._len_hdf = len(self.this_idx)

    def __len__(self):
        return self._len_hdf - self.config.maxlen

    def set_location_and_edge_matrix(self):
        # get lat longs
        wsmeta_ordered = self.hdf["wsmeta_ordered"][...]
        lats = [wsmeta_ordered[i] for i in range(0, len(wsmeta_ordered), 3)]
        long = [wsmeta_ordered[i] for i in range(1, len(wsmeta_ordered), 3)]

        # define location matrix
        self.loc_mat = torch.from_numpy(wsmeta_ordered.reshape[-1, 3].astype(np.float32))

        # define edge matrix
        edge_mat = np.ones((len(lats), len(lats)), dtype = np.float32)
        for i, (lat1, lon1) in enumerate(zip(lats, long)):
            for j, (lat2, lon2) in enumerate(zip(lats[i+1:], long[i+1:])):
                d = convert_latlong_to_great_circle(
                    (lat1, lon1), (lat2, lon2)
                )
                d2 = 1 / math.sqrt(d)
                edge_mat[i, i+j+1] = d2
                edge_mat[i+j+1, i] = d2
        edge_mat /= np.sqrt(edge_mat + 0.001)

        self.edge_mat = torch.from_numpy(edge_mat.astype(np.float32))

    def norm_sequence(self, seq):
        seq[..., 0] /= 1
        seq[..., 1] /= 1
        seq[..., 2] /= 1
        seq[..., 3] /= 1
        seq[..., 4] /= 1
        seq[..., 5] /= 1
        seq[..., 6] /= 1
        seq[..., 7] /= 1
        seq[..., 8] /= 1
        seq[..., 9] /= 1
        seq[..., 10] /= 1
        seq[..., 11] /= 1
        seq[..., 12] /= 1
        seq[..., 13] /= 1
        seq[..., 14] /= 1
        seq[..., 15] /= 1
        seq[..., 16] /= 1
        seq[..., 17] /= 1
        return seq

    def get_index(self, index: str):
        idata = self.hdf[index]
        mask = idata["mask"][...]
        data = idata["data"][...].reshape(len(mask), 17)
        data = self.norm_sequence(data)
        return data, mask

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
        config = self.config
        local_idx = self.this_idx[index]
        df = np.ndarray((config.maxlen, 611, 17), dtype=np.float32)
        masks = np.ndarray((config.maxlen, 611))
        for i in range(config.maxlen):
            data, mask = self.get_index(str(local_idx + i))
            df[i, ...] = data
            masks[i, ...] = mask
        dflist = df.tolist()
        msklist = masks.tolist()

        del df
        del masks

        {
            "input": torch.from_numpy(np.asarray(dflist).astype(np.float32)),  # [T, N, F]
            "locations": self.loc_mat,  # [T, N, F]
            "node_mask": torch.from_numpy(np.asarray(msklist)).long(), # [T, N, N]
            "edge_matrix": self.edge_mat, # [N, N]
            "month_ids",  # [T,]
            "day_ids",  # [T,]
            "hour_ids",  # [T,]

        }

class DatasetConfig:
    maxlen = None # what is the maximum length of sequence to return
    hdf_fpath = None
    index = None
    wsid_meta = None

    def __init__(self, **kwargs):
        self.attrs = []
        for k,v in kwargs.items():
            setattr(self, k, v)
            self.attrs.append(k)

    def __repr__(self):
        return "---- DATASET CONFIGURATION ----\n" + \
            "\n".join([f"{k}\t{getattr(self, k)}" for k in list(set([
                "maxlen",
                "hdf_fpath",
                "index",
                "wsid_meta",
            ] + self.attrs))
        ]) + "\n"
