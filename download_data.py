import pandas as pd
import numpy as np

import requests
import zlib
import os

from multiprocessing import  Pool
from tqdm import tqdm

def _df_split_apply(tup_arg):
    split_ind, df_split = tup_arg
    return (split_ind, df_split.apply(download_image, axis=1))

def df_multiprocess(df, processes, parts):
    splits = np.array_split(df, parts)
    print(len(splits), "parts", len(splits[0]), "images per part, on", processes, "processes")
    # Keep track of splits with an index, results are fastest first, not in order
    pool_data = [(split_ind, df_split) for split_ind, df_split in enumerate(splits)]

    pbar = tqdm(desc="Downloading", total=len(df), position=0)

    results = []
    with Pool(processes) as pool:
        for i, result in enumerate(pool.imap_unordered(_df_split_apply, pool_data, 2)):
            results.append(result)
            pbar.update(len(result[1]))
    
    results = sorted(results, key=lambda x:x[0])
    results = pd.concat([split[1] for split in results])

    pbar.close()
    return results

# Don't download image, just check with a HEAD request
# Can use this in _df_split_apply instead of download_image to get HTTP status codes
def check_download(row):
    # Unique name based on url
    fname = "%s/%s" % (row['folder'], (zlib.crc32(row['url'].encode('utf-8')) & 0xffffffff))
    try:
        # not all sites will support HEAD
        response = requests.head(row['url'], stream=False, timeout=2) 
    except:
        # log errors later, set error as 408 timeout
        row['status'] = 408
        return row

    row['status'] = response.status_code
    if response.ok:
        row['file'] = fname
    return row

def download_image(row):
    # Unique name based on url
    fname = "%s/%s" % (row['folder'], (zlib.crc32(row['url'].encode('utf-8')) & 0xffffffff))

    # Skip Already downloaded, retry others later
    if os.path.isfile(fname):
        row['status'] = 200
        row['file'] = fname
        return row

    try:
        # use smaller timeout to skip errors, but can result in failed downloads
        response = requests.get(row['url'], stream=False, timeout=5) 
    except:
        # log errors later, set error as 408 timeout
        row['status'] = 408
        return row
    
    row['status'] = response.status_code
    if response.ok:
        try:
            with open(fname, 'wb') as out_file:
                # some sites respond with gzip transport encoding
                response.raw.decode_content = True
                out_file.write(response.content)
        except:
            # This is if it times out during a download or decode
            row['status'] = 408
            return row
        row['file'] = fname
    return row

def open_tsv(fname, folder):
    print("Opening %s Data File..." % fname)
    df = pd.read_csv(fname, sep='\t', names=["caption","url"], usecols=range(1,2))
    df['folder'] = folder
    df['status'] = None
    df['file'] = None
    #Shuffling here can help when restarting, but then results are out of order.
    #np.random.shuffle(df.values)
    print("Processing", len(df), " Images:")
    return df

# number of processes in the pool can be larger than cores
num_processes = 64
# how many images per chunk per process
images_per_part = 4

# Validation:
df = open_tsv("Validation_GCC-1.1.0-Validation.tsv","validation")
df = df_multiprocess(df=df, processes=num_processes, parts=int(len(df)/images_per_part))
df.to_csv("downloaded_validation_report.tsv", sep='\t', header=False, index=False)

# Training:
df = open_tsv("Train_GCC-training.tsv","training")
df = df_multiprocess(df=df, processes=num_processes, parts=int(len(df)/images_per_part))
df.to_csv("downloaded_training_report.tsv", sep='\t', header=False, index=False)
