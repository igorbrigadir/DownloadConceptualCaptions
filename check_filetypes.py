from tqdm import tqdm
import filetype
import pandas as pd

def open_tsv_for_filetypecheck(fname):
    df = pd.read_csv(fname, sep='\t', names=["url", "folder", "status", "fname"], usecols=range(0,4))
    df['type'] = None
    return df

def check_filetypes(df):
    '''
    A reasonably fast way of detemining which files are images.
    Takes ~ 6.9E-4s / image on the validation set (real	0m6.861s)
    '''
    image_count = 0
    with tqdm(desc="Checking filetypes ...", total=len(df.index)) as pbar:
        for i in df.index:
            if df.iloc[i]['status'] == 200:
                ft = filetype.guess(df.iloc[i]['fname'])
                if ft is not None:
                    df.at[i, 'type'] = ft.mime
                    if 'image' in ft.mime:
                        image_count += 1
                else:
                    df.at[i, 'type'] = None
            else:
                df.at[i, 'type'] = None
            pbar.update(1)
    print("Found {} images / {} records".format(image_count, len(df.index)))
    return df

df = open_tsv_for_filetypecheck("downloaded_validation_report.tsv")
df = check_filetypes(df)
df.to_csv("filetypes-downloaded_validation_report.tsv", sep='\t', header=False, index=False)
