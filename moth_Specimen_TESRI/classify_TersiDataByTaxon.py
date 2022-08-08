import shutil
from pathlib import Path
import re
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from skimage import io 
from PIL import Image
import argparse
import time


# ===============================================================================================
parser = argparse.ArgumentParser(
    description='Just for copy imgs equally to different directorys '
)

# parser.add_argument('--save_dir', default='model/Unsup_rmbg')
# parser.add_argument("--basedir", '-dir', default='',
#                     type=str, help='where the data directory to predict')
parser.add_argument('--start_idx', default=0, type=int,
                    help="Manual epoch number (useful on restarts)")
parser.add_argument('--end_idx', default=-1, type=int,
                    help="Manual epoch number (useful on restarts)")
parser.add_argument('--sep', default=5000, type=int,
                    help="slice range for data")

args = parser.parse_args()
print(args)
# ===============================================================================================


dir_meta = Path('./meta')
dir_tersi_for_check = Path('./tersi_for_check')


dir_tersi = Path('../moth_thermal_project/data/data_resize_cropped/tersi_imgs_forYY_cropped256_paddingbg')
paths_tersi = list(dir_tersi.glob('*.png'))
print(f'data size in {dir_tersi.name} : {len(paths_tersi):,d}')

df_tesri_filelist_taxon = pd.read_csv(dir_meta.joinpath('tesri_filelist_taxon.csv'), index_col=0)
data_size = len(df_tesri_filelist_taxon)
print(f'data size of df_tesri_filelist_taxon : {data_size:,d}')

sep = args.sep
start = args.start_idx
end = data_size if args.end_idx == -1 else args.end_idx

df_tesri_select = df_tesri_filelist_taxon[start:end].copy()
data_size_select =  len(df_tesri_select)
print(f'data size delected : {data_size_select :,d}')

def copyfile(file:str):
    file_ = file + '_cropped.png' if not file.endswith('.png') else file

    try:
        shutil.copyfile(dir_tersi.joinpath(file_), 
                        dir_.joinpath(file_))
        print(f'\t{dir_.joinpath(file_)} copied\t\t\t\t\t', end='\r')
                        
    except shutil.SameFileError:
        print("\nSource and destination represents the same file.")
    except Exception as e:
        print('\n',e)
    
start_time = time.time()
start_ = 0
for i in range(int(data_size/sep + 1)):
    i_ = int(i*sep)
    end_ = start_ + sep
    
    if i_<end and i_>=start:
        # print(i_)
        
        dir_ = f'dir_{start_}_{end_}'
        dir_ = dir_tersi_for_check.joinpath(dir_)
        dir_.mkdir(exist_ok=True, parents=True)
        print(f'{dir_} maked')

        for idx, (_, rows) in enumerate(df_tesri_filelist_taxon[start_:end_].iterrows()) :
            family, subfamily, genus, species, file = rows
            file_ = file + '_cropped.png'
            # print(idx, file_)
            copyfile(file_)
            
            time_passed = time.time()-start_time
            print(f"i: {idx+1:4d}, {100*(idx)/data_size_select:.2f}% \
                | Time : {time_passed//60:.0f}m, {time_passed%60:.0f}s\t", end='\r')
            
    start_ = end_

start_ ,end_
start, end
print('Finished')

df_tesri_select