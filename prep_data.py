"""single script to prepare the dataset
28.09.2020 - @yashbonde"""

import os
import logging
import requests
import subprocess
import pandas as pd
from glob import glob
from tqdm import tqdm
from io import StringIO
from dateparser import parse
from datetime import datetime
from argparse import ArgumentParser

from datetime import timedelta, date

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

logger = logging.getLogger(f"prepare_data_{str(datetime.now())}.log")

COLUMNS = [
    ("date", None),
    ("hour", None),
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


def open_csv(f):
    # how I got "iso8859_1"
    # https://stackoverflow.com/questions/21504319/python-3-csv-file-giving-unicodedecodeerror-utf-8-codec-cant-decode-byte-err
    with open(f, "r", encoding="iso8859_1") as d:
        meta = d.readlines()[:8]

    p = {}
    keys = ["region", "name", 'estate', 'wsid', 'lat', 'long', 'elev', 'start']
    for i, m in enumerate(meta):
        k, v = m.strip().split(";")
        if "," in v:
            v = float(v.replace(",", "."))
        elif "-" in v:
            v = parse(v)
        p[keys[i]] = v

    with open(f, "r", encoding="iso8859_1") as d:
        df = pd.read_csv(
            StringIO("".join(d.readlines()[8:])),
            sep=";",
            encoding="iso8859_1"
        )
    return df, p


def download_file(url, folder):
    local_filename = folder + "/" +url.split('/')[-1]
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192)):
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk:
                f.write(chunk)
    return local_filename



if __name__ == "__main__":
    BAD_ROW_SUM = -169983.0

    args = ArgumentParser()
    args.add_argument('-d', action='store_true', help = "download all files")
    args.add_argument('-x', action='store_true', help="extract zip files")
    args.add_argument('-p', action='store_true', help = "parse dataset")
    args.add_argument('--folder', type = str, default="INMET", help = "folder to store all the years data")
    args = args.parse_args()

    urls = [f"https://portal.inmet.gov.br/uploads/dadoshistoricos/{x}.zip" for x in range(2000, 2021)]
    # print(urls)
    os.makedirs(args.folder, exist_ok=True)

    if args.d:
        for u in urls:
            print("fetching url:", u)
            if os.path.exists("./" + u.split("/")[-1]):
                print("Skipping")
                continue
            subprocess.run(["wget", u, ""])        
        subprocess.run(["mv", "*.zip", args.folder + "/"])

        for f in glob(args.folder + "/*.zip"):
            subprocess.run(["unzip", f, "-d", args.folder])

    if args.p:
        consol = {} # <date: [<active_wsids>]>
        for f in glob(args.folder + "/20*/*.CSV"):
            # go over each file
            print("Opening:", f)

            _, reg, wsname, wsid, city, start_date, _, end_date = f.replace(".CSV", "").split("_")

            start_date = date(*list(map(int, start_date.split("-")))[::-1])
            end_date = date(*list(map(int, end_date.split("-")))[::-1])

            # df, meta = open_csv(f)

            # df = df.dropna(axis=1)
            # for col in df.columns[2:]:
            #     v = getattr(df, col).values
            #     v = [float(str(x).replace(",", ".")) for x in v]
            #     setattr(df, col, v)

            # brs = BAD_ROW_SUM; idx = 0
            # while brs == BAD_ROW_SUM:
            #     brs = sum(df.loc[idx][2:])
            #     idx += 1

            # df.columns = COL_NAMES

            for single_date in daterange(start_date, end_date):
                d = single_date.strftime("%Y-%m-%d")
                consol.setdefault(d, [])
                consol[d].append(wsid)

            

            # for li in range(idx, df.shape[0]):
            #     dt = f"{df.loc[idx].date}_{df.loc[idx].date}"

            #     consol.setdefault(dt, [])
            #     consol[dt].append(wsid)

        for k,v in consol.items():
            consol[k] = ",".join(v)

        data = {
            "dates": sorted(list(consol.keys())),
            "wsids": sorted(list(consol.values()))
        }

        pd.DataFrame(data).to_csv(args.folder + "/consol.csv")

    print("completed, exiting")



    
