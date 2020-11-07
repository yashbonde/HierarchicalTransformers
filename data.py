"""
===============
How to do data?
The entire weather dataset cannot be loaded completely in memory
so this has to be either
A. iterative datatset where we yield samples [no]
B. we can store samples in a seperate file and load when required [no]
C. read a particular line from file lazily (using linecache) [no]
D. use a dedicated system like hdf5

Option D sounds the simplest as it is already is used to save the scientific
data like earth observation, etc. It took very long time to built the HDF5
takes 7:30 hours to have the final dump.

Part #1
-------
So HDF5 is short for heirarchical data format (v)5. HDF5 uses a directory
style method to store datasets, consider this example

/wsmeta_ordered
/datetime
1/data
1/meta
2/data
2/meta

Here 1,2... are groups and data, meta, wsmeta_ordered, datetime are
data. This is also the structure used in this HDF5 dump where each integer
group is the index from unified index.

Part #2
-------
This step started as creating a script but now I have added a notebook
with step by step blocks on how to get the final data that you want.
goto: _notebooks/prepare_hdf5.ipynb

Part #3
-------
Data in the HDF5 dump is not normalised and thus has to be normalised at
each call for fetching, this is because I had not come up with a strategy on
how to normalise this at the time of dumping. Each sequence is normalised
using `norm_sequence` function.

All normalised values are given in `norm_sequence`

# 15.09.2020 - @yashbonde
"""

import h5py
import math
import numpy as np
from tabulate import tabulate

import torch
from torch.utils.data import Dataset

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = torch.cuda.current_device()

# ---- norm methods ---- #
def log_norm(x, thresh = 4500, eps = 1e-5):
    x[x > thresh] = thresh
    x[x <= 0] = eps
    return np.nan_to_num(np.log(x + eps))

normalisation_functions = {
    "temp": lambda x: (x - 23.70349134402726) / 5.546263797529582,
    "max_temp": lambda x: (x - 23.70349134402726) / 5.546263797529582,
    "min_temp": lambda x: (x - 23.70349134402726) / 5.546263797529582,
    
    "dew_point_temp": lambda x: (x - 17.228775847494408) / 4.874964913785063,
    "min_dew": lambda x: (x - 17.228775847494408) / 4.874964913785063,
    "max_dew": lambda x: (x - 17.228775847494408) / 4.874964913785063,
    
    "pressure": lambda x: (x - 965.715544383282) / 37.57273866463611,
    "min_pressure": lambda x: (x - 965.715544383282) / 37.57273866463611,
    "max_pressure": lambda x: (x - 965.715544383282) / 37.57273866463611,
    
    "wind_speed": lambda x: (x - 3.9103860481733657) / 2.820015788831639,
    "wind_gust": lambda x: (x - 3.9103860481733657) / 2.820015788831639,
    "wind_direction": lambda x: (x - 180) / 180,
    
    "humidity": lambda x: x / 90,
    "max_humidity": lambda x: x / 90,
    "min_humidity": lambda x: x / 90,
    
    "radiation": lambda x: log_norm(x, 4500),
    "total_precipitation": lambda x: log_norm(x, 30),
}

keys = [
    'total_precipitation',
    'pressure',
    'max_pressure',
    'min_pressure',
    'radiation',
    'temp',
    'dew_point_temp',
    'max_temp',
    'min_temp',
    'max_dew',
    'min_dew',
    'max_humidity',
    'min_humidity',
    'humidity',
    'wind_direction',
    'wind_gust',
    'wind_speed'
]

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
        self.mode = mode
        
        self.num_wsid = self.edge_mat.shape[2]

    def __len__(self):
        return self._len_hdf - self.config.maxlen

    def __repr__(self):
        return (f"<WeatherDataset {len(self)} samples in" 
        f" '{self.mode}' mode, {self.num_wsid} nodes>")


    def set_location_and_edge_matrix(self):
        # get lat longs
        wsmeta_ordered = self.hdf["wsid_meta"][...]
        lats = [wsmeta_ordered[i] for i in range(0, len(wsmeta_ordered), 3)]
        long = [wsmeta_ordered[i] for i in range(1, len(wsmeta_ordered), 3)]

        # define location matrix
        self.loc_mat = torch.from_numpy(wsmeta_ordered.reshape(1, 1, -1, 3).astype(np.float32))

        # define edge matrix
        edge_mat = np.ones((len(lats), len(lats)), dtype = np.float32)
        for i, (lat1, lon1) in enumerate(zip(lats, long)):
            for j, (lat2, lon2) in enumerate(zip(lats[i+1:], long[i+1:])):
                d = convert_latlong_to_great_circle((lat1, lon1), (lat2, lon2))
                d2 = 1 / math.sqrt(d)
                edge_mat[i, i+j+1] = d2
                edge_mat[i+j+1, i] = d2
        edge_mat /= np.sqrt(edge_mat + 0.001)
        
        N, N = edge_mat.shape
        edge_mat = edge_mat.reshape(1, 1, N, N)

        self.edge_mat = torch.from_numpy(edge_mat.astype(np.float32))

    def norm_sequence(self, seq):
        for i in range(len(keys)):
            # get key and then apply that particular normalisation function
            seq[:, i] = normalisation_functions[ keys[i] ](seq[:, i])
        return seq

    def get_index(self, index: str):
        # observational data
        idata = self.hdf[index]
        mask = idata["mask"][...]
        data = idata["data"][...].reshape(len(mask), 17)
        data = self.norm_sequence(data)

        # time data
        time = self.hdf["datetime"][int(index)]

        return data, mask, time

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
        df = np.ndarray((config.maxlen, self.num_wsid, 17), dtype=np.float32)
        masks = np.ndarray((config.maxlen, self.num_wsid))
        months = np.ndarray((config.maxlen,))
        days = np.ndarray((config.maxlen,))
        hours = np.ndarray((config.maxlen,))
        for i in range(config.maxlen):
            data, mask, (mon, day, hrs) = self.get_index(str(local_idx + i))
            df[i, ...] = data
            masks[i, ...] = mask
            months[i] = mon
            days[i] = day
            hours[i] = hrs
        dflist = df.tolist()
        msklist = masks.tolist()
        months_list = months.tolist()
        days_list = days.tolist()
        hours_list = hours.tolist()

        T_S = df.shape[0] // config.seqlen
        S = config.seqlen
        N, F = df.shape[1:]

        # save mem?
        del df
        del masks
        del months, days, hours

        return {
            "input": torch.from_numpy(
                np.asarray(dflist).astype(np.float32).reshape(T_S, S, N, F)
            ).to(DEVICE),  # [T, N, F]

            "node_mask": torch.from_numpy(
                np.asarray(msklist).reshape(T_S, S, N)
            ).long().to(DEVICE), # [T_S, S, N]

            "month_ids": torch.from_numpy(
                np.asarray(months_list).reshape(T_S, S)
            ).long().to(DEVICE),  # [T_S, S]
            "day_ids": torch.from_numpy(
                np.asarray(days_list).reshape(T_S, S)
            ).long().to(DEVICE),  # [T_S, S]
            "hour_ids": torch.from_numpy(
                np.asarray(hours_list).reshape(T_S, S)
            ).long().to(DEVICE),  # [T_S, S]
        }

class DatasetConfig:
    maxlen = None # what is the maximum length of sequence to return
    hdf_fpath = None
    index = None
    seqlen = None

    def __init__(self, **kwargs):
        self.attrs = [
            "maxlen",
            "hdf_fpath",
            "index",
            "seqlen"
        ]
        for k,v in kwargs.items():
            setattr(self, k, v)
            self.attrs.append(k)

    def __repr__(self):
        return "---- DATASET CONFIGURATION ----\n" + \
             tabulate(
                 [(k,getattr(self, k)) for k in list(set(self.attrs))],
                 headers=["key", "value"],
                 tablefmt="psql"
            )


# ---- For Demo ---- #
class DummyDataset(Dataset):
    def __init__(self, config):
        assert config.maxlen % config.seqlen == 0, \
            f"cannot break {config.maxlen} long sequence into equal {config.seqlen} pieces"
        self.config = config

        self.input_sequences = np.array([
            np.sin(np.linspace(0, np.pi * (i + 1), config.n_samples)).tolist()
            for i in range(config.n_features * config.n_nodes)
        ]).T.reshape(-1, config.n_nodes, config.n_features) # [n_samples, N, F]

        # print("--->", self.input_sequences.shape)

        self.node_mask = np.ones(shape = (config.n_samples, config.n_nodes))
        for i in range(self.node_mask.shape[0]):
            if np.random.random() > 0.75:
                mask_node = np.random.randint(
                    low=0,
                    high=self.node_mask.shape[1],
                    size=(
                        np.random.randint(
                            low = 0, high = 4
                        )
                    )
                )
                self.node_mask[i, mask_node] = 0
        
        self.loc_mat = torch.from_numpy(np.random.randn(config.n_nodes, 3).astype(np.float32))
        self.edge_mat = torch.from_numpy(np.random.randn(config.n_nodes, config.n_nodes).astype(np.float32))

        # create ordered time
        hrs, day, mon = [], [], []
        hr_cntr, day_cntr, mon_cntr = 0, 0, 0
        for i in range(config.n_samples):
            # print(hr_cntr, day_cntr, mon_cntr)
            hrs.append(hr_cntr); day.append(day_cntr); mon.append(mon_cntr)
            hr_cntr += 1
            if hr_cntr == 24: # every 24 hours a day passes in Africa
                hr_cntr = 0
                day_cntr += 1

            if day_cntr and day_cntr % 16 == 0 and hr_cntr == 0:  # every 16 days a month passes in Africa
                mon_cntr += 1
            
            if day_cntr == 32:
                day_cntr = 0

            if mon_cntr == 13: # every 12 months a year passes in Africa
                mon_cntr = 0
        
        # dammit, I really wanted to make original meme
        self.hrs = np.asarray(hrs).astype(np.int32)
        self.day = np.asarray(day).astype(np.int32)
        self.day[self.day == 31] = 30
        self.mon = np.asarray(mon).astype(np.int32)

    def __len__(self):
        return self.input_sequences.shape[0] - self.config.maxlen

    # since these two things are constants that do not change ever, we
    # can pre-fetch them so useless memory is not utilised copying shit
    def get_edge_matrix(self):
        return self.edge_mat

    def get_location_matrix(self):
        return self.loc_mat

    # main function to get the data
    def __getitem__(self, index):
        config = self.config

        # get all the data points first
        # print(index, index + config.maxlen)
        curr_seq = self.input_sequences[index: index+config.maxlen] # done to get shapes
        msklist = self.node_mask[index: index+config.maxlen].tolist()
        months_list = self.mon[index: index+config.maxlen].tolist()
        days_list = self.day[index: index+config.maxlen].tolist()
        hours_list = self.hrs[index: index+config.maxlen].tolist()

        # now convert datapoints to sequence of data points
        T_S = curr_seq.shape[0] // config.seqlen
        S = config.seqlen
        N, F = curr_seq.shape[1:]

        curr_seq = curr_seq.tolist() # finally convert to list aye

        return {
            "input": torch.from_numpy(
                np.asarray(curr_seq).astype(np.float32).reshape(T_S, S, N, F)
            ),  # [T // S, S, N, F]
            "node_mask": torch.from_numpy(
                np.asarray(msklist).reshape(T_S, S, N)
            ).long(), # [T // S, S, N]
            
            "month_ids": torch.from_numpy(np.asarray(months_list).reshape(T_S, S)).long(),  # [T // S, S]
            "day_ids": torch.from_numpy(np.asarray(days_list).reshape(T_S, S)).long(),  # [T // S, S]
            "hour_ids": torch.from_numpy(np.asarray(hours_list).reshape(T_S, S)).long(),  # [T // S, S]
        }


class DummyDatasetConfig():
    def __init__(
        self,
        n_samples = 100,
        n_nodes = 30,
        n_features = 5,
        maxlen = 10,
        seqlen = 5
    ):
        self.n_samples = n_samples
        self.n_nodes = n_nodes
        self.n_features = n_features
        self.maxlen = maxlen
        self.seqlen = seqlen

# # --- Actual --- #
# if __name__ == "__main__":
#     config = DatasetConfig(
#         maxlen = 10,
#         hdf_fpath="/Users/yashbonde/Desktop/AI/vv2/_notebooks/weatherGiga2.hdf5",
#         index="/Users/yashbonde/Desktop/AI/vv2/_notebooks/test_idx.txt",
#     )

#     print(config)

#     wd = WeatherDataset(config)
#     print(wd)
#     print("EdgeMatrix:", wd.edge_mat.shape)
#     print("LocationMatrix:", wd.loc_mat.shape)
#     for i in range(5):
#         print(f'----- INDEX {i} -----')
#         print({k:(v.size(), v.dtype) for k,v in wd[i].items()})

#     from torch.utils.data import DataLoader

#     print()
#     for x in DataLoader(wd, batch_size = 32, shuffle = True):
#         print({k: (v.size(), v.dtype) for k, v in x.items()})
#         break

# # ----- Demo ----- #
# if __name__ == "__main__":
#     config = DummyDatasetConfig()
#     wd = DummyDataset(config)
#     # print(wd.mon, wd.day, wd.hrs)
#     print(wd[0])
