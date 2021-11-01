import os
import sys
import glob
from pathlib import Path
from typing import Counter
import numpy as np
from numpy.lib.npyio import save
import skimage.io as io
from skimage import data, img_as_float
from skimage import exposure
from PIL import Image



dir_target = Path('data/label_waiting_postprocess/mask_waitinting_for_posrprocess/for_removebg')
pathes_target = list(dir_target.rglob('*.png')) 
print(len(pathes_target))

dir_save = dir_target.joinpath('AddContrast')
dir_save.mkdir(exist_ok=True, parents=True)

for idx, path in enumerate(pathes_target):
    name = path.stem
    img = io.imread(path)
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)

    io.imsave(dir_save.joinpath(name + '.png'), img_adapteq)
    print(idx, name)
