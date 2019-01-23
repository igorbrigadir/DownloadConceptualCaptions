import pandas as pd
import numpy as np

import requests
import zlib
import os

from multiprocessing import Pool
from tqdm import tqdm

import magic #pip install python-magic

headers = {
    #'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36',
    'User-Agent':'Googlebot-Image/1.0', # Pretend to be googlebot
    'X-Forwarded-For': '64.18.15.200'
}

def _df_split_apply(tup_arg):
    split_ind, df_split, func = tup_arg
    return (split_ind, df_split.apply(func, axis=1))

def df_multiprocess(df, processes, parts, func):
    splits = np.array_split(df, parts)
    print(len(splits), "parts", len(splits[0]), "images per part, on", processes, "processes")
    # Keep track of splits with an index, results are fastest first, not in order
    pool_data = [(split_ind, df_split, func) for split_ind, df_split in enumerate(splits)]

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

# Unique name based on url
def _file_name(row):
    return "%s/%s" % (row['folder'], (zlib.crc32(row['url'].encode('utf-8')) & 0xffffffff))

# For checking mimetypes separately without download
def check_mimetype(row):
    if os.path.isfile(str(row['file'])):
        row['mimetype'] = magic.from_file(row['file'], mime=True)
        row['size'] = os.stat(row['file']).st_size
    return row

# Don't download image, just check with a HEAD request, can't resume.
# Can use this instead of download_image to get HTTP status codes.
def check_download(row):
    fname = _file_name(row)
    try:
        # not all sites will support HEAD
        response = requests.head(row['url'], stream=False, timeout=5, allow_redirects=True, headers=headers)
        row['status'] = response.status_code
        row['headers'] = dict(response.headers)
    except:
        # log errors later, set error as 408 timeout
        row['status'] = 408
        return row
 
    if response.ok:
        row['file'] = fname
    return row

def download_image(row):
    fname = _file_name(row)

    # Skip Already downloaded, retry others later
    if os.path.isfile(fname):
        row['status'] = 200
        row['file'] = fname
        return row

    try:
        # use smaller timeout to skip errors, but can result in failed downloads
        response = requests.get(row['url'], stream=False, timeout=10, allow_redirects=True, headers=headers)
        row['status'] = response.status_code
        row['headers'] = dict(response.headers)
    except Exception as e:
        # log errors later, set error as 408 timeout
        row['status'] = 408
        return row
    
    if response.ok:
        try:
            with open(fname, 'wb') as out_file:
                # some sites respond with gzip transport encoding
                response.raw.decode_content = True
                out_file.write(response.content)

            row['mimetype'] = magic.from_file(row['file'], mime=True)
            row['size'] = os.stat(row['file']).st_size
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
    df['mimetype'] = None
    df['size'] = None
    df['file'] = None
    df['headers'] = None
    #Shuffling here can help when restarting, but then results are out of order.
    #np.random.shuffle(df.values)
    print("Processing", len(df), " Images:")
    return df

# number of processes in the pool can be larger than cores
num_processes = 96
# how many images per chunk per process
images_per_part = 2

# Validation download:
df = open_tsv("Validation_GCC-1.1.0-Validation.tsv","validation")
df = df_multiprocess(df=df, processes=num_processes, parts=int(len(df)/images_per_part), func=download_image)
df.to_csv("downloaded_validation_report.tsv.gz", compression='gzip', sep='\t', header=False, index=False)

# Training download:
df = open_tsv("Train_GCC-training.tsv","training")
df = df_multiprocess(df=df, processes=num_processes, parts=int(len(df)/images_per_part), func=download_image)
df.to_csv("downloaded_training_report.tsv.gz", compression='gzip', sep='\t', header=False, index=False)

# Validation mime type check only:
#print('Checking Downloaded Files:')
#df = pd.read_csv("downloaded_validation_report.tsv.gz", compression='gzip', sep='\t', names=["url","folder","status","mimetype","size","file","headers"])
#df = df_multiprocess(df=df, processes=num_processes, parts=int(len(df)/images_per_part), func=check_mimetype)
#df.to_csv("downloaded_validation_report.tsv.gz", compression='gzip', sep='\t', header=False, index=False)

# Training HEAD request check only:
#df = open_tsv("Train_GCC-training.tsv","training")
#df = df_multiprocess(df=df, processes=num_processes, parts=int(len(df)/images_per_part), func=check_download)
#df.to_csv("downloaded_training_report.tsv.gz", compression='gzip', sep='\t', header=False, index=False)
