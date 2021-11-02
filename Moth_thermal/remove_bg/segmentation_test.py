from skimage import exposure
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import time
from pathlib import Path
from PIL import Image
import skimage.io as io
from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
# %matplotlib inline


base_dir = '../crop/CARS_cropped256_120up/'
files_ = glob.glob(base_dir + '**/*.png', recursive=True)

path = files_[0]

# img = img_as_float(astronaut()[::2, ::2])
img = io.imread(path)
# Image.fromarray(img).show()

segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=20000)
segments_slic = slic(img, n_segments=1000, compactness=10, sigma=0)
segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
gradient = sobel(rgb2gray(img))
segments_watershed = watershed(gradient, markers=250, compactness=0.001)

print(f'Felzenszwalb number of segments: {len(np.unique(segments_fz))}')
print(f'SLIC number of segments: {len(np.unique(segments_slic))}')
print(f'Quickshift number of segments: {len(np.unique(segments_quick))}')

fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

ax[0, 0].imshow(mark_boundaries(img, segments_fz))
ax[0, 0].set_title("Felzenszwalbs's method")
ax[0, 1].imshow(mark_boundaries(img, segments_slic))
ax[0, 1].set_title('SLIC')
ax[1, 0].imshow(mark_boundaries(img, segments_quick))
ax[1, 0].set_title('Quickshift')
ax[1, 1].imshow(mark_boundaries(img, segments_watershed))
ax[1, 1].set_title('Compact watershed')

for a in ax.ravel():
    a.set_axis_off()

plt.tight_layout()
# plt.show()

save_dir = os.path.join('tmp', 'segmentation')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

plt.savefig(os.path.join(save_dir, 'segmentation.jpg'))
# ======================================================================================================================


file = 'spot_color_failure'
dir_target = Path(f'data/data_for_Unsup_rmbg/{file}')

dir_save = Path(f'data/tmp/{file}_AddContrast')
dir_save.mkdir(parents=True, exist_ok=True)

c = 0
for path in dir_target.iterdir():
    name = path.stem
    img = io.imread(path)
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.01)
    io.imsave(dir_save.joinpath(name + '.png'), img_adapteq)
    print(c, name)
    c += 1
