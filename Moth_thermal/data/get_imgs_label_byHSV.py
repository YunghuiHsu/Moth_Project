import os
import sys
from pathlib import Path
from datetime import datetime
import math
import shutil
import numpy as np
import pandas as pd
from PIL import Image
import skimage.io as io
from skimage import color
from sklearn.cluster import KMeans, Birch
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import umap
from sklearn.decomposition import PCA

from plotly import graph_objects as go
import matplotlib.pyplot as plt
# %matplotlib inline
# import seaborn as sns
# sns.set_style('dark')
# %config Completer.use_jedi = False

colors = ['#000080', '#00008b', '#0000ff', '#006400', '#008000', '#008080',
          '#00bfff', '#00ced1', '#00fa9a', '#00ff00', '#00ff7f', '#191970',
          '#1e90ff', '#2e8b57', '#2f4f4f', '#32cd32', '#3cb371', '#4682b4',
          '#483d8b', '#556b2f', '#696969', '#7b68ee', '#7f0000', '#7f007f',
          '#7fffd4', '#808000', '#8b0000', '#8b4513', '#8fbc8f', '#90ee90',
          '#9370db', '#9400d3', '#9932cc', '#9acd32', '#a0522d', '#a9a9a9',
          '#adff2f', '#afeeee', '#b03060', '#b0e0e6', '#bc8f8f', '#cd853f',
          '#daa520', '#db7093', '#dc143c', '#dda0dd', '#deb887', '#ee82ee',
          '#f08080', '#f0e68c', '#f4a460', '#fa8072', '#ff0000', '#ff00ff',
          '#ff1493', '#ff6347', '#ff8c00', '#ffc0cb', '#ffd700', '#ffe4c4']


# ====================================================================================================
# Sample pixels on moth specimen and calculate HSV values
# ====================================================================================================

dir_imgs = Path('./data_for_Sup_train/imgs/')
# dir_imgs = Path('../../data/origin')

pathes_imgs = list(dir_imgs.glob('*.png'))
print(f'Number of imgs : {len(pathes_imgs)}')


# -------------------------------------------------------------------------------------------
# Set sample location by coordinate
# image size: 256x256
# center point of image : (h*0.5, w*0.5) == 128,128

w_samples = np.linspace(15, 240, 25).astype('uint8')
h_samples = np.linspace(50, 215, 20).astype('uint8')

samples_grid = {}
for x, p_x in enumerate(w_samples):
    if p_x > 108 and p_x < 148:
        continue
    for y, p_y in enumerate(h_samples):
        # if p_x > 122 and p_x <134
        point = f'{p_x}_{p_y}'
        # 　note:　shape of image from io.imread is (h, w). So the coordonate should be y,x
        samples_grid[point] = p_y, p_x
len(samples_grid)


dir_tmp = Path('./tmp/crop_test')
dir_tmp.mkdir(exist_ok=True, parents=True)


def sample_img(img: np.ndarray, sample_loc: dict, size: int = 3, save_crop: bool = False) -> dict:
    if save_crop:
        img_sample = img.copy()
    for loc_, loc_coord in sample_loc.items():
        y, x = loc_coord
        img_crop = img[y-size:y+size, x-size:x+size]

        img_crop_hsv = color.rgb2hsv(img_crop)
        h, s, v = [img_crop_hsv[..., c] for c in [0, 1, 2]]
        wing_sample_hsv[loc_] = [h.mean(), s.mean(), v.mean()]

        if save_crop:
            img_sample[y-size:y+size, x-size:x+size] = 0
    if save_crop:
        # for check sample location
        io.imsave(dir_tmp.joinpath(f'{name}_sample.jpg'), img_sample)
    return wing_sample_hsv


# moth_hsv_ = dict()
# print(f'Samplimg HSV by grids')
# for idx, path in enumerate(pathes_imgs):
#     # if idx % 5 == 0:
#     name = path.stem.split('_cropped')[0]
#     img = io.imread(path)
#     wing_sample_hsv = dict()
#     wing_sample_hsv = sample_img(
#         img, samples_grid, size=3, save_crop=False)
#     moth_hsv_[name] = wing_sample_hsv
#     print(f'\t{idx:04d}, {name:50s}', end='\r')
#     # if idx == 600:
#     #     break


def moth_hsv_to_df(dict_hsv: dict) -> pd.DataFrame:
    df = (pd.DataFrame(dict_hsv).T
          .reset_index()
          .rename(columns={'index': 'Name'}))
    df_new = df['Name']
    for column in df.columns.values[1:]:
        df_split = pd.DataFrame(df[column].tolist(), columns=[
                                f'{column}_h', f'{column}_s', f'{column}_v'])
        df_new = pd.concat([df_new, df_split], axis=1)
    return df_new


# df_moth_hsv_ = moth_hsv_to_df(moth_hsv_)
save_path_moth_hsv_ = f'../../data/imgs_{len(pathes_imgs)}_hsv_samples_grid.csv'
# df_moth_hsv_.to_csv(save_path_moth_hsv_)

# ---------------------------------------------------------------------------
# filter samples_grid
df = pd.read_csv(save_path_moth_hsv_, index_col=0)

# df_std_h = df.iloc[:, 1::3].std()    # slice cols from col:1 (Hue)
df_std_s = df.iloc[:, 2::3].std()    # slice cols from col:2 (saturation)
# df_std_v = df.iloc[:, 3::3].std()    # slice cols from col:3 (brightness)
df_std = df_std_s

# filter grid where std(hue) less variant
threshold = np.quantile(df_std, 0.5)
loc_filter = df_std[df_std > threshold].index.values
print(f'\nsample points from {len(samples_grid)} to {len(loc_filter)}')


samples_grid_wings = {}
for obj in loc_filter:
    x, y, _ = obj.split('_')
    samples_grid_wings[f'{x}_{y}'] = int(y), int(x)
# len(samples_grid_wings)

np.save('./samples_grid_wings.npy', samples_grid_wings)
samples_grid_wings_ = np.load('./samples_grid_wings.npy', allow_pickle=True)
samples_grid_wings = [rows for idx, rows in np.ndenumerate(samples_grid_wings_)][0]
print(f'\n{samples_grid_wings} loaded')

moth_hsv = dict()
print(f'Samplimg HSV on Wings')
for idx, path in enumerate(pathes_imgs):
    if idx % 17 == 0:
        name = path.stem.split('_cropped')[0]
        img = io.imread(path)

        wing_sample_hsv = dict()
        wing_sample_hsv = sample_img(img, samples_grid_wings, size=3, save_crop=True)
    moth_hsv[name] = wing_sample_hsv

    print(f'\t{idx:04d}, {name:50s}', end='\r')
    if idx == 600:
        break

df_new = moth_hsv_to_df(moth_hsv)
df_new.to_csv(
    f'./imgs_{len(pathes_imgs)}_hsv_samples_wing.csv')

# ====================================================================================================
# Clustering moth specimen images depends HSV values
# ====================================================================================================

# Decomposition
print('\nStart Decomposition')
df_new = pd.read_csv(
    f'./imgs_{len(pathes_imgs)}_hsv_samples_wing.csv', index_col=0)

X = df_new.iloc[:, 1:]
Z = StandardScaler().fit_transform(X)    # Normalization,  u=0, std=1

Y = df_new.iloc[:, 0]

# pca = PCA(n_components=3)
# pca = PCA(0.9)
# embedding = pca.fit_transform(X)
# embedding.shape
# ratio = pca.explained_variance_ratio_
# pd.DataFrame(ratio).cumsum().plot()
# plt.savefig('tmp/ratio.jpg')
n_neighbors = int(len(pathes_imgs)*0.05)

dimension_reductor = umap.UMAP(
    n_components=3, n_neighbors=n_neighbors, random_state=42)
embedding = dimension_reductor.fit_transform(Z)

# ------------------------------------------------------------------------------------------------------
# Clustering performance evaluation
# ====================================================================================================
print('\nClustering performance evaluation')

time_ = datetime.now().strftime("%y%m%d_%H%M")
# dir_save = Path('../tmp/cluster_test')
dir_save = Path(f'./tmp/cluster_test_{time_}')
dir_save.mkdir(exist_ok=True, parents=True)

n_cluster = range(2, 21)

sil_scores = []
cal_scores = []
dav_scores = []
for c in n_cluster:
    cls = Birch(n_clusters=c)
    labels = cls.fit_predict(embedding)

    sil_score = metrics.silhouette_score(embedding, labels, metric='euclidean')
    sil_scores.append(sil_score)
    cal_score = metrics.calinski_harabasz_score(embedding, labels)
    cal_scores.append(cal_score)
    dav_score = metrics.davies_bouldin_score(embedding, labels)
    dav_scores.append(dav_score)


def normalize(scores: np.array) -> np.array:
    # 0 - 1
    z = (np.array(scores) - np.min(scores)) / (np.max(scores) - np.min(scores))
    return z


weighted_scores = (normalize(sil_scores) +
                   normalize(cal_scores) + normalize(-np.array(dav_scores))) / 3

metrics_ = ['Silhouette_score', 'Calinski_harabasz_score',
            'Davies_bouldin_score', 'Weighted_score']
scores = [sil_scores, cal_scores, dav_scores, weighted_scores]
for metric, score in zip(metrics_, scores):
    pd.DataFrame(score, index=n_cluster, columns=[metric]).plot()
    plt.savefig(dir_save.joinpath(f'Birch_cluster_{metric}.jpg'))
    plt.close()
    print(f'\t{str(dir_save.joinpath(f"Birch_cluster_{metric}.jpg"))} saved')

weighted_scores[:5] = 0  # cluster must > 10
idx_ = np.argmax(weighted_scores)
best_n_cluster = n_cluster[idx_]

print(f'\tGet the best clusting number : {best_n_cluster}')

# ------------------------------------------------------------------------------------------------------
# Clustering
# ------------------------------------------------------------------------------------------------------

print('\nStart Clustering')
# cls = OPTICS(min_samples=5, xi=0.35)

cls = Birch(
    # threshold=1.2, n_clusters=None,
    n_clusters=best_n_cluster
)
cls_ids = cls.fit_predict(embedding)

# kmeans = KMeans(n_clusters=7, random_state=0).fit(embedding)
# cls_ids = kmeans.labels_

print('\t', 'Clustering Number :', np.unique(cls_ids).size)
print('\t', 'Clustering id and size : ',
      np.unique(cls_ids, return_counts=True))

df_label = pd.concat(
    [df_new['Name'], pd.DataFrame(cls_ids, columns=['label'])], axis=1)
path_save_label = f'./imgs_label_byHSV.csv'
df_label.to_csv(path_save_label)
print(f'{path_save_label} saved')

df_label = pd.read_csv(path_save_label, index_col=0)

# ====================================================================================================
# Visualiztion clustering result
# ====================================================================================================

# # plotting 3d figure
fig = go.Figure(data=[go.Scatter3d(
    x=embedding[:, 0],
    y=embedding[:, 1],
    z=embedding[:, 2],
    mode='markers',
    marker=dict(
        size=3,
        color=np.array(colors)[cls_ids],
        opacity=0.8,
        line=dict(
            color='black',
            width=0
        )
    ),
    text=cls_ids
)])
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.write_html(dir_save.joinpath('clusters_3d.html'))
print(f'\n{str(dir_save.joinpath("clusters_3d.html"))} saved')

# plotting Thumbnail of imgs by clustering
print('\nPlotting Thumbnail of imgs by clustering')
index_id = {}
for id in df_label.label.unique():
    indexes = df_label[df_label.label == id].index.values
    index_id[id] = indexes


def plot_imgs_cluster(imgs: np.array, id: int, index: list, df_label: pd.DataFrame):
    ncols = 20
    nrows = math.ceil(len(index)/ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(ncols, nrows))
    title = f"Cluster_{id}"
    fig.suptitle(title, fontsize=int(24*nrows/ncols), y=0.92)
    for i, ax in enumerate(axes.flatten()):
        ax.set_axis_off()
        if i < len(index):
            # ax.set_title(f"cluster : {list(cluster_index.keys())[i]}")
            ax.imshow(imgs[i])
        else:
            continue
    fig.savefig(dir_save.joinpath(
        f'Birch_cluster{id}_{len(index)}.jpg'), bbox_inches="tight")
    plt.close()

    print(f'\t{str(dir_save.joinpath(f"cluster{id}_{len(index)}.jpg"))} saved')


for id, index in index_id.items():
    print(f'\tClustering id : {id}, size : {len(index)}')
    imgs = []
    for idx in index:
        name = df_label.Name[idx]
        path = dir_imgs.joinpath(name + '_cropped.png')
        img = io.imread(path)
        imgs.append(img)
    imgs_numpy = np.asarray(imgs)
    plot_imgs_cluster(imgs=imgs_numpy, id=id, index=index, df_label=df_label)


# for id in np.unique(cls_ids):
#     dir_save.joinpath(str(id)).mkdir(exist_ok=True, parents=True)

# for idx, path in enumerate(pathes_imgs):
#     name = path.stem
#     print(idx, name, end='\t\t\r')
#     cls_id = cls_ids[idx]

#     shutil.copyfile(dir_imgs.joinpath(name + '.png'),
#                     dir_save.joinpath(str(cls_id), name + '.jpg'))
