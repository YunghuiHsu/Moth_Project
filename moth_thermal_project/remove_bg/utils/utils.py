import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import torch


def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()


def plt_result(img, title, save_name=None, img_save=None, img_save_dir=None, fontsize=25, show=False):
    '''
    img, title -> list()
    '''
    fig = plt.figure(figsize=(20, 20), dpi=400)

    for i in range(len(img)):
        ax = fig.add_subplot('1'+str(len(img))+str(i+1))
        ax.set_title(title[i], fontsize=fontsize)
        plt.axis('off')  # hide the number line on the x,y borders
        #ax.title.set_text("\nD(intra)={0}, D(inter)={1}".format(1,2))
        if img[-1] is not 3:
            ax.imshow(img[i], cmap='gray')
        else:
            ax.imshow(img[i])
#     save_to = OUTPUT_DIR+'CHECKING/'
#     if not os.path.exists(save_to):
#             os.makedirs(save_to)
#     fig.savefig(save_to + save_name +'.png', dpi=100, format='png',bbox_inches='tight' )
    if not show == True:
        plt.close()
    else:
        plt.show()


#     for i in range(len(img_save)):

#         save_to = OUTPUT_DIR+img_save_dir[i]+'/'
#         if not os.path.exists(save_to):
#             os.makedirs(save_to)
#         imageio.imwrite(save_to + save_name +'.png', img_save[i])
    return fig


def plt_learning_curve(train_loss_noted, valid_loss_noted, title='', sub='', st='', ed=''):
    fig = plt.figure()  # figsize=(4, 4), dpi= 100)
    if not st == '':
        plt.plot(np.arange(
            len(train_loss_noted[st:ed])), train_loss_noted[st:ed], 'b', label='train')
        plt.plot(np.arange(
            len(valid_loss_noted[st:ed])), valid_loss_noted[st:ed], 'r', label='valid')
    else:
        plt.plot(np.arange(len(train_loss_noted)),
                 train_loss_noted, 'b', label='train')
        plt.plot(np.arange(len(valid_loss_noted)),
                 valid_loss_noted, 'r', label='valid')
    plt.title(title)
    plt.suptitle(sub, fontsize=10, color='gray')
    plt.xlabel('epoch', fontsize=10)
    plt.ylabel('loss', fontsize=10)
    plt.legend(loc='best')


#     plt.annotate(annot,
#             xy=(0, 0), xytext=(len(train_loss_noted), 0),
#             xycoords=('axes fraction', 'figure fraction'),
#             textcoords='offset points',
#             size=8, color = 'gray',  ha='left', va='bottom')
    plt.close()
    return fig


def early_stop(valid: float, best_value: float,  trigger_times: int = 0, patience: int = 10, metric: str = 'min') -> int:
    '''
    valid:: validation value for loss or accuracy or dice score.
    best_value:: best value of validation value.
    metric:: 'min' or 'max'  for minimize(loss), else maxmize(accuracy ,dice score).
    patience::　if trigger, number of waiting depends on patience.
    '''

    if metric == 'min':
        cond = valid > best_value  # minimize(loss)
    elif metric == 'max':
        cond = valid < best_value  # maxmize(accuracy ,dice score)

    if cond:
        trigger_times += 1
        print('\nEarly stopping trigger times:', trigger_times)
    else:
        trigger_times = 0
        print('\ntrigger times reset:', trigger_times)

    # if trigger_times >= patience:
    #     print('\nTriger Early Stop!')
    return trigger_times


def get_data_attribute(img_paths: list) -> pd.DataFrame:
    names_imgs = [path.stem.split('_cropped')[0] for path in img_paths]

    df = pd.DataFrame(names_imgs, columns=['Name'])
    cond1 = df.Name.str.contains('CARS')
    cond2 = df.Name.str.contains('SJTT')
    index_CARS = df[cond1].index.values
    index_SJTT = df[cond2].index.values
    df['Source'] = np.nan
    df.iloc[index_CARS, 1] = 'CARS'
    df.iloc[index_SJTT, 1] = 'SJTT'

    presufix = (df.Name.str.split('CARS', expand=True).loc[:, 0]
                .str.split('SJTT', expand=True).loc[:, 0]
                .str.rstrip('_')
                .str.replace('\d', ''))
    df['Family'] = presufix
    index_none = df[df.Family == ''].index.values
    df.Family[index_none] = 'None'
    df['Source_Family'] = df.Family + '_' + df.Source
    df.to_csv('data_attribute.csv', index=False)

    return df


def get_masks_contour(masks: np.ndarray, iterations: int = 5, weighted: float = 2.0) -> torch.Tensor:
    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]
    assert masks.ndim == 3, f'Shape of mask must be (h,w) or (batch, h, w), mask.shape : {masks.shape}'
    assert masks.dtype == 'uint8', f'dtype of mask need to be "uint8", masks.dtype : {masks.dtype}.\nuse .astype("uint8") convert dtype'

    kernel_ELLIPSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(3, 3))

    masks_contour = []
    for mask_ in masks:
        mask_dilate = cv2.dilate(mask_, kernel_ELLIPSE, iterations=6)
        mask_erode = cv2.erode(mask_, kernel_ELLIPSE, iterations=iterations)
        mask_contour = mask_dilate - mask_erode
        masks_contour.append(mask_contour)

    masks_contour_tensor = torch.tensor(masks_contour, dtype=torch.long)

    return masks_contour_tensor
