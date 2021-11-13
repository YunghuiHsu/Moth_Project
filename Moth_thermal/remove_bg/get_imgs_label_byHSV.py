import os
import sys
from pathlib import Path
import glob
import time
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

dir_imgs = Path('../data/data_for_Sup_train/imgs/')
pathes_imgs = list(dir_imgs.glob('*.png'))
print(f'Number of imgs : {len(pathes_imgs)}')


# Set sample location by coordinate
# image size: 256x256
# center point of image : (h*0.5, w*0.5) == 128,128

h, w = 255, 255
sample_forewing = [(0.025, 0.10), (0.05, 0.15), (0.10, 0.20),
                   (0.125, 0.25), (0.15, 0.30), (0.05, 0.30)]
sample_backwing = [(0.025, 0.10), (0.05, 0.15), (0.10, 0.20),
                   (0.05, 0.20), (0.10, 0.15), (0.05, 0.25)]

wing_sample = dict()
for side in ['l', 'r']:
    for part in ['fw', 'bw']:
        sample_ = sample_forewing if part == 'fw' else sample_backwing
        for point in [0, 1, 2, 3, 4, 5]:
            sample = sample_[point]
            loc_ = f'{side}_{part}{point}'
            if side == 'l' and part == 'fw':
                wing_sample[loc_] = (
                    int(h*0.5 - h * sample[0]), int(w*0.5 - w * sample[1]))
            elif side == 'l' and part == 'bw':
                wing_sample[loc_] = (
                    int(h*0.5 + h * sample[0]), int(w*0.5 - w * sample[1]))
            elif side == 'r' and part == 'fw':
                wing_sample[loc_] = (
                    int(h*0.5 - h * sample[0]), int(w*0.5 + w * sample[1]))
            elif side == 'r' and part == 'bw':
                wing_sample[loc_] = (
                    int(h*0.5 + h * sample[0]), int(w*0.5 + w * sample[1]))

# add background sample point
wing_sample['bg_bot'] = (int(h*0.5 - h*0.40), int(w*0.5 + w*0.00))
wing_sample['bg_top'] = (int(h*0.5 + h*0.40), int(w*0.5 + w*0.00))


dir_tmp = Path('tmp/crop_test')
dir_tmp.mkdir(exist_ok=True, parents=True)


def sample_img(img: np.ndarray, wing_sample: dict, size: int = 5) -> dict:
    # img_sample = img.copy()
    for loc_, loc_coord in wing_sample.items():
        y, x = loc_coord
        img_crop = img[y-size:y+size, x-size:x+size]

        img_crop_hsv = color.rgb2hsv(img_crop)
        h, s, v = [img_crop_hsv[..., c] for c in [0, 1, 2]]
        wing_sample_hsv[loc_] = [round(h.mean(), 5), round(
            s.mean(), 5), round(v.mean(), 5)]

        # img_sample[y-size:y+size, x-size:x+size] = 0

    # io.imsave(dir_tmp.joinpath(f'{name}_sample.jpg'), img_sample)
    return wing_sample_hsv


moth_hsv = dict()
print(f'Samplimg HSV on Wings')
for idx, path in enumerate(pathes_imgs):
    # if idx % 3 == 0:
    name = path.stem.split('_cropped')[0] + '_cropped'
    img = io.imread(path)

    wing_sample_hsv = dict()
    wing_sample_hsv = sample_img(img, wing_sample, size=3)
    moth_hsv[name] = wing_sample_hsv

    # io.imsave(dir_tmp.joinpath(f'{name}.jpg'), img)
    print('\t', idx, name, end='\t\t\r')

    # if idx == 600:
    #     break

df = (pd.DataFrame(moth_hsv).T
      .reset_index()
      .rename(columns={'index': 'Name'}))
df_new = df['Name']
for column in df.columns.values[1:]:
    df_split = pd.DataFrame(df[column].tolist(), columns=[
                            f'{column}_h', f'{column}_s', f'{column}_v'])
    df_new = pd.concat([df_new, df_split], axis=1)
# df_new.to_csv(f'imgs_{len(pathes_imgs)}_hsv.csv')

# ====================================================================================================
# Clustering moth specimen images depends HSV values
# ====================================================================================================

# Decomposition
print('\nStart Decomposition')
# df_new = pd.read_csv(f'imgs_{len(pathes_imgs)}_hsv.csv', index_col=0)

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
n_neighbors = int(len(pathes_imgs)*0.1)
dimension_reductor = umap.UMAP(
    n_components=3, n_neighbors=n_neighbors, random_state=42)
embedding = dimension_reductor.fit_transform(Z)

# ------------------------------------------------------------------------------------------------------
# Clustering performance evaluation
# ====================================================================================================
print('\nClustering performance evaluation')

dir_save = Path('tmp/cluster_test')
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

weighted_scores[:3] = 0  # cluster must > 3
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
df_label.to_csv(f'imgs_label_byHSV.csv')
print(f'imgs_label_byHSV.csv saved')


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
    ncols = 12
    nrows = math.ceil(len(index)/ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(16, 16*nrows/ncols))
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
        path = dir_imgs.joinpath(name + '.png')
        img = io.imread(path)
        imgs.append(img)
    imgs_numpy = np.asarray(imgs)
    plot_imgs_cluster(imgs=imgs_numpy, id=id, index=index, df_label=df_label)

# get imgs by cld_id
for id in np.unique(cls_ids):
    dir_save.joinpath(str(id)).mkdir(exist_ok=True, parents=True)

for idx, path in enumerate(pathes_imgs):
    name = path.stem
    print(idx, name, end='\t\t\r')
    cls_id = cls_ids[idx]
    # if cls_id == 1 or cls_id == 2:
    shutil.copyfile(dir_imgs.joinpath(name + '.png'),
                    dir_save.joinpath(str(cls_id), name + '.png'))
