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
    description='Assign imgs to different directorys by taxon'
)

parser.add_argument('--file', default='meta/MC_tesri_tagnull_imgpath.csv')
parser.add_argument('--dir_save', '-save', default='./tersi_for_check')
parser.add_argument("--dir_target", '-dir', default='./tesri_img',
                    type=str, help='where the data directory to predict')
parser.add_argument('--start_idx', '-s', default=0, type=int,
                    help="Manual epoch number (useful on restarts)")
parser.add_argument('--end_idx', '-e',default=-1, type=int,
                    help="Manual epoch number (useful on restarts)")
parser.add_argument('--img_suffix', '-suffix',type=str, default='.jpg')
# parser.add_argument('--sep', default=5000, type=int,
#                     help="slice range for data")

args = parser.parse_args()
print(args)
# ===============================================================================================
def copyfile(file:str, dir_)->None:
    
    assert file.split('.')[-1].lower() in ['jpg', 'jpeg', 'png']
    try:
        shutil.copyfile(dir_target.joinpath(file), 
                        dir_.joinpath(file))
                        
    except shutil.SameFileError:
        print("\nSource and destination represents the same file.")
    except Exception as err:
        print('\n',err)


dir_save = Path(args.dir_save)
suffix = args.img_suffix


dir_target = Path(args.dir_target)
paths_target = list(dir_target.glob('*'))
paths_target = [path for path in paths_target 
               if path.suffix.lower() in ['.jpg', '.jpeg', '.png'] ]
print(f'data size in {dir_target.name} : {len(paths_target):,d}')

file = args.file
df_file = pd.read_csv(file, index_col=0)
data_size = len(df_file)
print(f'data size of df_file : {data_size:,d}')

# replace columns of df
col_dict = {}
for column in df_file.columns.values:
    if column.lower() == 'family':
        col_dict['family'] = column
    elif column.lower() == 'species' :
        col_dict['sp'] = column
    elif column.lower().endswith('id') :
        col_dict['id'] = column
    # elif column == 'associatedMedia' or column.lower().endswith('path'):
    #     col_dict['path'] = column
df_file = df_file[list(col_dict.values())]

start = args.start_idx
end = data_size if args.end_idx == -1 else args.end_idx

df_file_select = df_file[start:end].copy()
print(f'data size delected : {len(df_file_select) :,d}')


start_time = time.time()
for idx, (_, rows) in enumerate(df_file_select.iterrows()) :
    family,  species, id = rows
    file = id + suffix
    
    dir_ = dir_save.joinpath(family,  species)
    if not dir_.exists() :
        dir_.mkdir(exist_ok=True,parents=True)
        
    copyfile(file, dir_)

    time_passed = time.time()-start_time
    
    info = f'[{idx+1:4d}/{len(df_file_select):,d}], {100*(idx)/len(df_file_select):.1f}%'
    info += f' | Time : {time_passed//60:.0f}m, {time_passed%60:.0f}s'
    info += f' | {dir_.joinpath(file)} copied\t\t'
    
    print(info, end='\r')



print('\nFinished')

# ------------------------------------------------------------------------------------------------
# assign by sep number
# sep = args.sep
# for i in range(int(data_size/sep + 1)):
#     i_ = int(i*sep)
#     end_ = start_ + sep
    
#     if i_<end and i_>=start:
#         # print(i_)
        
#         dir_ = f'dir_{start_}_{end_}'
#         dir_ = dir_save.joinpath(dir_)
#         dir_.mkdir(exist_ok=True, parents=True)
#         print(f'{dir_} maked')

#         for idx, (_, rows) in enumerate(df_file[start_:end_].iterrows()) :
#             family, subfamily, genus, species, file = rows
#             file = file + '_cropped.png'
#             # print(idx, file)
#             copyfile(file)
            
#             time_passed = time.time()-start_time
#             print(f"i: {idx+1:4d}, {100*(idx)/len(df_file_select):.2f}% \
#                 | Time : {time_passed//60:.0f}m, {time_passed%60:.0f}s\t", end='\r')
            
#     start_ = end_