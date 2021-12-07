
import os
import sys
import glob
from pathlib import Path

import numpy as np
import skimage.io as io
from PIL import Image
import math
import matplotlib.pyplot as plt

# =================================================================================================

dir_benchmarks = Path('../data/data_for_Sup_predict/benchmarks')
pathes_benchmarks = list(dir_benchmarks.glob('*.png'))
dir_benchmarks_rmbg = Path('tmp/imgs_rmbg')
paths_benchmarks_rmbg = list(dir_benchmarks_rmbg.glob('*.png'))
sys.getsizeof(pathes_benchmarks_rmbg)

dir_mask = Path('../data/data_for_Sup_train/masks_211105')
imgs_mask = list(dir_mask.glob('*.png'))
names_benchmarks = [path.stem for path in dir_benchmarks]
paths_benchmarks_mask = [
    path for path in imgs_mask if path.stem in names_benchmarks]
len(paths_benchmarks_mask)

dir_BatchAugment = Path('tmp/BatchAugment')

# dir_1st : MultiImg、SingleImg、Random;  dir_2st : 'Predict_rmbg', 'Predict_mask'
method_BatchAugment = ['Random', 'SingleImg', 'MultiImg']
dir_BatchAugment_rmbg = [dir_BatchAugment.joinpath(dir, 'Predict_rmbg') for dir in method_BatchAugment]
paths_BatchAugment_rmbg = [list(dir.glob('*.png')) for dir in dir_BatchAugment_rmbg]

dir_BatchAugment_mask = [dir_BatchAugment.joinpath(dir, 'Predict_mask') for dir in method_BatchAugment]
paths_BatchAugment_mask = [list(dir.glob('*.png')) for dir in dir_BatchAugment_mask]

len(paths_BatchAugment_rmbg)

# plot_ccomparative figs
#             | origin/true,  'RandomBatch','SingleImgBatch','MultiImgBatch'
# img         |       o               -              -              -
# (img_rmbg)  |      (o)              o              o              o
# (mask)      |      (o)              o              o              o


dir_savefig = dir_BatchAugment.joinpath('plot_ccomparative_figs')
dir_savefig.mkdir(exist_ok=True, parents=True)

def plot_comparative_figs(idx:int, name:str, img_origin:np.ndarray, img_rmbgs:list, img_masks:list, ncols:int=4, nrows:int=3):
    factor = 3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                            figsize=(ncols*factor, int(nrows*factor)))
    title = f"{name}"
    fig.suptitle(title, fontsize=int(16*factor*nrows/ncols), y=1.05)
    type_BatchAugment = ['Origin', 'Random', 'SingleImg', 'MultiImg']
    for r, rows in enumerate(axes):
        for c, ax in enumerate(rows):
            ax.set_axis_off()
            if r == 0:
                ax.set_title(f"{type_BatchAugment[c]}", fontsize=int(10*factor*nrows/ncols))
                if c == 0:
                    ax.set_axis_on()
                    ax.set_xticks([]), ax.set_yticks([])
                    ax.set_ylabel('origin', fontsize=int(10*factor*nrows/ncols))
                    ax.imshow(img_origin)
                else:
                    continue
            if r == 1:
                if c==0:
                    ax.set_axis_on()
                    ax.set_xticks([]), ax.set_yticks([])
                    ax.set_ylabel('rmbg', fontsize=int(10*factor*nrows/ncols))
                path_rmbg = img_rmbgs[c]
                img_rmbg = io.imread(path_rmbg)
                ax.imshow(img_rmbg)
            if r == 2:
                if c==0:
                    ax.set_axis_on()
                    ax.set_xticks([]), ax.set_yticks([])
                    ax.set_ylabel('mask', fontsize=int(10*factor*nrows/ncols))
                path_mask = img_masks[c]
                img_mask = io.imread(path_mask)
                ax.imshow(img_mask, cmap='gray')
    fig.tight_layout()
    plt.subplots_adjust(
        # left=0.0, bottom=0.0, right=0.0, top=0.0, 
        wspace=0.003, hspace=0.005
        )
    fig.savefig(dir_savefig.joinpath(
        f'{idx:03d}_{name}_BatchArg.jpg'), bbox_inches="tight")
    plt.close()



for idx, path in enumerate(pathes_benchmarks):
    name = path.stem
    path_ = dir_benchmarks.joinpath(name + '.png')
    img_origin = io.imread(path_)
    img_rmbgs = [paths_benchmarks_rmbg[idx]] + [paths_BatchAugment_rmbg[i][idx] for i in [0, 1, 2]]
    img_masks = [paths_benchmarks_mask[idx]] + [paths_BatchAugment_mask[i][idx] for i in [0, 1, 2]]
    plot_comparative_figs(idx, name, img_origin, img_rmbgs, img_masks)
    
    print(idx, name, 'saved')
    # if idx==5: break
