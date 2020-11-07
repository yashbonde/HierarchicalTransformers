"""single script to prepare the dataset
28.09.2020 - @yashbonde

Note: there are things that might seem super unintuitive particularly in
args.c # construct. That is """

import re
import os
import json
import logging
import requests
import subprocess
import pandas as pd
from glob import glob
from tqdm import trange
from io import StringIO
from dateparser import parse
from datetime import datetime
from argparse import ArgumentParser

from datetime import timedelta, date
def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

# logger = logging.getLogger(f"prepare_data_{str(datetime.now())}.log")
logging.basicConfig(level = logging.INFO)

COLUMNS = [
    ("total_precipitation", "mm"),
    ("pressure", "mB"),
    ("max_pressure", "mB"),
    ("min_pressure", "mB"),
    ("radiation", "KJ/m^2"),
    ("temp", "C"),
    ("dew_point_temp", "C"),
    ("max_temp", "C"),
    ("min_temp", "C"),
    ("max_dew", "C"),
    ("min_dew", "C"),
    ("max_humidity", "percentage"),
    ("min_humidity", "percentage"),
    ("humidity", "percentage"),
    ("wind_direction", "deg"),
    ("wind_gust", "m/s"),
    ("wind_speed", "m/s")
]
COL_NAMES = [x[0] for x in COLUMNS]
UNITS = [x[1] for x in COLUMNS]
BAD_ROW_SUM = 0.0

KEYS = [
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

def open_csv(f, wsid):
    with open(f, "r", encoding="iso8859_1") as d:
        df = pd.read_csv(
            StringIO("".join(d.readlines()[8:]).replace("-9999", "0")),
            sep=";",
            encoding="iso8859_1"
        )
    
    # this can also drop main columns so specify the main column to drop
    # df = df.dropna(axis = 1)
    df = df.drop("Unnamed: 19", axis = 1)
    local_col_names = [f"{wsid}_{c}" for c in COL_NAMES]
    df.columns = ["date", "hour"] + local_col_names
    for col in local_col_names[2:]:
        setattr(df, col, getattr(df, col).apply(lambda x: float(str(x).replace(",", "."))))
        
    return df

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument('-d', action='store_true', help = "download all files")
    args.add_argument('-x', action='store_true', help="extract zip files")
    args.add_argument('-p', action='store_true', help = "parse dataset")
    args.add_argument('-c', action='store_true', help="compile dataset")
    args.add_argument('--folder', type = str, default="INMET", help = "folder to store all the years data")
    args.add_argument('--from_year', default = 2007, type = int, help = "dump is created from this year")
    args = args.parse_args()

    os.makedirs(args.folder, exist_ok=True)

    if args.d:
        urls = [f"https://portal.inmet.gov.br/uploads/dadoshistoricos/{x}.zip" for x in range(max(2000, args.from_year), 2021)]
        for u in urls:
            logging.info("fetching url: {u}")
            if os.path.exists("./" + u.split("/")[-1]):
                logging.info("Skipping")
                continue
            subprocess.run(["wget", u, ""])        
        subprocess.run(["mv", "*.zip", args.folder])

    if args.x:
        for f in glob(args.folder + "/*.zip"):
            subprocess.run(["unzip", f, "-d", args.folder])

    if args.p:
        # consol data
        meta_master = {}  # weather station meta
        consol = {} # <date: [<active_wsids>]>
        all_files = glob(args.folder + "/20*/*.CSV", ncols = 100)
        for i, f in zip(trange(len(all_files)), all_files):
            # go over each file
            start_date, end_date = re.findall(r"\d+-\d+-\d{4}", f)
            wsid = re.findall(r"[A-Z]\d{3}", f)[0]

            start_date = date(*list(map(int, start_date.split("-")))[::-1])
            end_date = date(*list(map(int, end_date.split("-")))[::-1])

            for single_date in daterange(start_date, end_date):
                d = single_date.strftime("%Y-%m-%d")
                consol.setdefault(d, [])
                consol[d].append(wsid)

            # add to weather station metas
            meta, _ = open_csv(f, m=True, d=False)
            if meta["wsid"] in meta_master:
                continue
            meta_master[meta.pop("wsid")] = meta

        for k,v in consol.items():
            consol[k] = ",".join(v)

        data = {
            "dates": sorted(list(consol.keys())),
            "wsids": sorted(list(consol.values()))
        }

        logging.info(f"Saving: {args.folder + '/consol.csv'}")
        pd.DataFrame(data).to_csv(args.folder + "/consol.csv")
        logging.info("Saving: {args.folder + '/wsid_meta.csv'}")
        pd.DataFrame(meta_master).to_csv(args.folder + "/wsid_meta.csv")


    # this is the most time comsuming part, construct the final dataset
    if args.c:
#         all_files = glob(args.folder + "/20*/*.CSV")

        # get all the dfs
#         logging.info(f"Get all dfs. Number of files: {len(all_files)} ... this will take sometime.")
#         dfs = []
#         year_to_idx = {}
#         cntr = 0
#         for i, f in zip(trange(len(all_files)), all_files):
#             year = int(f.split("/")[1])
#             if year < args.from_year:
#                 continue

#             wsid = re.findall(r"[A-Z]\d{3}", f)[0]
#             try:
#                 dfs.append(open_csv(f, wsid))
#             except Exception as e:
#                 logging.info(f"Failed: {f} --> {e}")

#             year_to_idx.setdefault(year, [])
#             year_to_idx[year].append(cntr)
#             cntr += 1

#         # seperate the datasets
#         year_wise = {}
#         for k, v in year_to_idx.items():
#             arr = []
#             for i in v:
#                 arr.append(dfs[i])
#             year_wise[k] = arr

#         # create exhaustive jsons
#         logging.info(f"dumping exhaustive jsons ... might take some time. Number of years found: {len(year_wise)}")
#         os.makedirs(args.folder + '/target', exist_ok=True)
#         for year in year_wise:
#             for df in year_wise[year]:
#                 wsid = df.columns[5].split("_")[0]
                
#                 # since 2019 they started using a new format
#                 if year in [2019, 2020]:
#                     df.date = [x.replace("/", "-") for x in df.date.values]
#                     df.hour = [f"{x.split()[0][:2]}:{x.split()[0][2:]}" for x in df.hour.values]
#                 df.to_json(args.folder + f"/target/{year}_{wsid}.json", orient = "columns")

#         # start making master
        all_files = glob(args.folder + "/target/*.json")
        logging.info(f"Found: {len(all_files)} files")

#         # first part is creating a unfied global index
#         logging.info("creating unified global index")
#         unified_idx = set()
#         for _, fpath in zip(trange(len(all_files)), all_files):
#             with open(fpath, 'r') as f:
#                 data = json.load(f)
#                 for i in data["date"]:
#                     d, h = data["date"][i], data["hour"][i]
#                     unified_idx.add(f"{d}T{h}")

#         unified_idx = sorted(list(set(unified_idx)))
#         logging.info(f"idxs: {len(unified_idx)}. top 10: {unified_idx[:10]}")

#         os.makedirs(args.folder + '/final', exist_ok=True)
#         with open(args.folder + "/final/index.json", "w") as f:
#             f.write(json.dumps({i: x for i, x in enumerate(unified_idx)}))
#         del unified_idx

        with open(args.folder + "/final/index.json", "r") as f:
            unified_idx = json.load(f)
        inv_idx = {unified_idx[x]: i for i, x in enumerate(unified_idx)}
        
        print(list(unified_idx.keys())[:10], list(inv_idx.keys())[:10])

        # next we group the data by wsid
        files_by_wsid = {}
        for f in all_files:
            wsid = f.split("/")[-1].split("_")[-1].split(".")[0]
            files_by_wsid.setdefault(wsid, [])
            files_by_wsid[wsid].append(f)
        files_by_wsid = {k: sorted(v) for k, v in files_by_wsid.items()}
        
        logging.info(f"Found: {len(files_by_wsid)} wsids")
        
        # open each file and get the global indexes for it.
        for _, (wsid, files) in zip(trange(len(files_by_wsid)), files_by_wsid.items()):
            wsid_data = {} # a unified dataset for all global_indexes for this wsid
            for fpath in files:
                with open(fpath, "r") as f:
                    data = json.load(f)
                    
                global_idx = [inv_idx[f"{data['date'][i]}T{data['hour'][i]}"] for i in data["date"]]
                data = pd.read_json(fpath)
                data = data.values[:, 2:]
                
                # fix dtype
                for idx in range(len(data)):
                    x0 = str(data[idx, 0]).replace(",", ".")
                    x1 = str(data[idx, 1]).replace(",", ".")
                    try:
                        data[idx, 0] = float(x0)
                    except:
                        data[idx, 0] = None
                        
                    try:
                        data[idx, 1] = float(x1)
                    except:
                        data[idx, 1] = None

                assert len(data) == len(global_idx)
                
                # merged
                merged = {gid:d.tolist() for gid, d in zip(global_idx, data)}
                wsid_data.update(merged)
            
            with open(args.folder + f"/final/{wsid}.json", "w") as f:
                f.write(json.dumps(wsid_data))

    logging.info("completed, exiting")

