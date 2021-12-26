from pathlib import Path
import numpy as np
import pandas as pd
from skimage import io
import time
import argparse
import urllib


# =======================================================================================================================
parser = argparse.ArgumentParser(
    description='Scrap for Moth Specimen photos from TESRI'
)
parser.add_argument('--dir_save', '-save',type=str, default="./tesri_img",
                    help='directory to save imgs')
parser.add_argument('--path_file','-p', type=str, default="meta/MC_tersi_tagnull_imgpath.csv",
                    help='file to get imgpath')
parser.add_argument('--start_idx', '-s', default=0, type=int,
                    help="Manual epoch number (useful on restarts)")
parser.add_argument('--end_idx', '-e', default=-1, type=int,
                    help="Manual epoch number (useful on restarts)")

args = parser.parse_args()
print(args)

# =======================================================================================================================

dir_save = Path(args.dir_save)
dir_meta = Path('./meta')
for dir in [dir_save, dir_meta]:
    dir.mkdir(exist_ok=True, parents=True)

path_file = Path(args.path_file)
print(f'{path_file} Loaded')
df_imgpath = pd.read_csv(path_file, index_col=0)
print(f'Total img_path : {len(df_imgpath):,d}')

start = args.start_idx
end = len(df_imgpath) if args.end_idx == -1 else args.end_idx
print(f'slice from [{start} : {end}]')

col_dict = {}
for column in df_imgpath.columns.values:
    if column.lower() == 'family':
        col_dict['family'] = column
    elif column.lower() == 'species' :
        col_dict['sp'] = column
    elif column.lower().endswith('id') :
        col_dict['id'] = column
    elif column == 'associatedMedia' or column.lower().endswith('path'):
        col_dict['path'] = column

df_imgpath = df_imgpath[list(col_dict.values())]
assert df_imgpath.columns.values.size == 4, f'There are problems about columns : {df_imgpath.columns.values}'

error_log =  dir_meta.joinpath(f'logging_download_fail_{path_file.stem}.txt')


start_time = time.time()
for idx, rows in df_imgpath.iloc[start: end].iterrows():
    family, sp, id, path = rows
    # print(f'idx : {idx:,d}, sp : {sp:20s}, id : {id:30s}')

    try:
        img = io.imread(path)
        io.imsave(dir_save.joinpath(id + '.jpg'), img)
        time.sleep(0.1)

        time_cost = time.time()-start_time
        info = f"====> Progress: [{idx}]/[{len(df_imgpath)}] | {100*idx/len(df_imgpath):.2f}%"
        info += f"| time: {time_cost//(60*60):2.0f}h{time_cost//60%60:2.0f}m{time_cost%60:2.0f}s"
        info += f"| {sp:s}, id : {id:s} saved\t\t\t\t"
        print(info, end='\r')

    except urllib.error.HTTPError as error_404:
        print(f'\n\t{error_404}')

        with open(error_log, 'a') as file:
            file.write(f'{idx}, {id}, {path}\n')
        print(f'\n\tgiveup : {idx} : {id}')

    except Exception as e:
        print(f'\t{e}')
        c = 0

        while c <= 10:
            print(f'\n\twaiting times : {c}')
            time.sleep(1)  # 強制等待
            img = io.imread(path)
            io.imsave(dir_save.joinpath(id + '.jpg'), img)
            c += 1
        else:
            with open(error_log, 'a') as file:
                file.write(f'{idx}, {id}, {path}\n')
            print(f'\n\tgiveup : {idx} : {id}')

print('\nFinished')
