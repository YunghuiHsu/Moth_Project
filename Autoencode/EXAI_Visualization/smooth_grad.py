import os, sys, glob, argparse, time
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable


path_root =  '/home/jovyan/Autoencoder'
path_EXAI =  "/home/jovyan/Autoencoder/EXAI_Visualization/SaliencyMap_SmoothGrad"
if not os.path.isdir(f'{path_EXAI}'):
    os.mkdir(f'{path_EXAI}')
    print(f'"{path_EXAI}" folder made!')
    
sys.path.append("..")  # 更改預測的import路徑，為了能讀取上一層級資料夾 MothScripts內的檔案
from MothScripts.resnet50_classifier import ResNet50_family_classifier
import skimage
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt

# ===============================================================================================

parser = argparse.ArgumentParser()

parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--epoch', type=int, default=100, help='epoch for smooth_grad')
parser.add_argument("--sigma_multiplier", default=0.4, type=float, help="sigma_multiplier for smooth_grad, Default: 0.4")
parser.add_argument("--setdevice", default="cuda:1", type=str, help="set which device to use, Default: cuda:1")
parser.add_argument('--size', default="128, 128", type=str, help='image size')
parser.add_argument("--start_iter", default=0, type=int, help="Manual iter number (useful on restarts)")
parser.add_argument("--data_start", default=0, type=int, help="Manual iter number (useful on restarts)")
parser.add_argument("--data_end", default=32262, type=int, help="Manual iter number (useful on restarts)")

args = parser.parse_args()
# args = parser.parse_args(args=[]) # in jupyter 

# =====load Model==========================================================================================
device = torch.device(args.setdevice if torch.cuda.is_available() else "cpu")

resnet50 = ResNet50_family_classifier(getDeepFeature=False)                        # getDeepFeature=False for get prediction only(else for feature map)

# Load pth

resnet_pretrained = torch.load(
    f"{path_root}/model/vsc_wgan/pretrained_fam_classification_resnet50_20210613.pth", 
    map_location=lambda storage, loc: storage.cpu()                                        
)


pretrained_dict, resnet50_dict = resnet_pretrained['model_state'], resnet50.state_dict()      # 載入權重參數
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in resnet50_dict}            # 更新pretrained model的key與value，使之與resnet50一致
resnet50_dict.update(pretrained_dict)                                                         # 將resnet50的權重更新為pretrained model
resnet50.load_state_dict(resnet50_dict)                                                       # 將pretrained的權重載入resnet50


# =====Prepare Dataset==========================================================================================
str_to_tuple = lambda x: tuple([int(xi) for xi in x.split(',')])

size = str_to_tuple(args.size)

class ImgDataset(Dataset):
    '''簡單用來抽樣看原始圖檔的Dataset
        X: ImagePath
        y: Imagelabel(Family, Genus ,Specie Name etc...)
    '''
    def __init__(self, X, y, size=(256, 256)):
        self.ImgNames  = np.asarray(X)                # 輸入的X 為影像完整路徑的list、y則為完整的科名list 
#         self.labels = np.asarray(y).astype('int32')
        self.labels = np.asarray(y)
        self.size = size
        self.to_tensor = transforms.ToTensor()        # 將取值範圍為[0, 255]的PIL.Image或形狀為[H, W, C]的numpy.ndarray，轉換成形狀為[C, H, W]，取值範圍是[0, 1.0]的

    def __getitem__(self, index):
        img_Name = self.ImgNames[index]
#         image = PILImage.open(img_Name)                #  PIL Image讀讀取影像的通道為(c,w,h)
        image = skimage.io.imread(img_Name)              #  skimage.io讀取影像的通道為(w,h,c)，to_tensor會自動調整  
        image = skimage.transform.resize(image, self.size)
        image = self.to_tensor(image).float()            # 根據檔名開啟影像。得到image.jpg的物件，並轉為Tensor
        label  = self.labels[index]
        img_Name = img_Name.split("/")[-1].split(".")[0]
        return image, label, img_Name

    def __len__(self):
        return len(self.labels)
    
    # help to get images for visualizing
    def getbatch(self, indices):
        images = []
        labels = []
        for index in indices:
            image, label = self.__getitem__(index)
            images.append(image)
            labels.append(label)
        return torch.stack(images), torch.tensor(labels)

metadata = f'{path_root}/meta/moth_meta_20210610.csv'
moth_meta = pd.read_csv(metadata)
# 取得檔案名稱加入欄位內
dataroot = "/home/jovyan/Autoencoder/wolrdwide_lepidoptera_yolov4_cropped_and_padded_20210610"
image_list = glob.glob(dataroot + '/*.jpg')  # 如果資料乾淨的話，可直接用 os.listdir(dataroot)
moth_meta['ImgPath'] = image_list

fam_dict = dict(
    zip(range(90), np.sort(moth_meta.Family.unique())
    ) 
) 


# 一次預讀取讀取所有影像 
s = args.data_start
e = args.data_end
X = moth_meta.ImgPath[s:e]
y = moth_meta.Family_encode[s:e]
img_set = ImgDataset(X=X, y=y)
class_names = moth_meta.Family  # Series


# =====Smooth grad==========================================================================================

def normalize(image):
    return (image - image.min()) / (image.max() - image.min())

def smooth_grad(x, y, model, epoch, param_sigma_multiplier):
    model.eval()
    #x = x.cuda().unsqueeze(0)

    mean = 0
    sigma = param_sigma_multiplier / (torch.max(x) - torch.min(x)).item()
    smooth = np.zeros((x).unsqueeze(0).size())
    for i in range(epoch):
        # call Variable to generate random noise
        model.zero_grad()
        noise = Variable(x.data.new(x.size()).normal_(mean, sigma**2))
        x_mod = (x + noise).unsqueeze(0).to(device)
        x_mod.requires_grad_()
        
        y_pred = model(x_mod)
#         y_pred, *_ = model(x_mod)

        loss_func = torch.nn.CrossEntropyLoss()
        loss = loss_func(y_pred, y.to(device).unsqueeze(0))
        loss.backward()

        # like the method in saliency map
        x_mod  = torch.clamp(x_mod.grad, min= 0.)           # relu
        smooth += x_mod.detach().cpu().data.numpy()
    smooth = normalize(smooth / epoch) # don't forget to normalize
    # smooth = smooth / epoch
    return smooth

model = resnet50.to(device)
if args.setdevice=="cpu":
    pin_memory=False
else:
    pin_memory=True

img_data_loader = DataLoader(img_set, batch_size=args.batchSize, shuffle=False, num_workers=2, pin_memory=False)

print(f"img_data_loader prepared, iter : {len(img_data_loader)}")

# =====Run Smooth_grad()==========================================================================================

start_time = time.time()

for i, (image, label, img_Name) in enumerate(img_data_loader):
    image, label  = image.to(device)[0], label.to(device)[0]
    smooth_ = smooth_grad(image, label, model, args.epoch , args.sigma_multiplier)[0]
#     smooth_ = (smooth_*255).astype('uint8')
    smooth_ = np.transpose(smooth_, (1,2,0))
#     smooth_grey = skimage.color.rgb2grey(smooth_)

#     plt.imshow(smooth_grey, cmap='Greys_r')
#     plt.imshow(smooth_grey)
    plt.imshow(smooth_)
    plt.axis('off')
    plt.savefig(f'{path_EXAI}/Smoothgrad_{img_Name[0]}.png', bbox_inches='tight', pad_inches=0.)
    plt.close();
    
    time_pass = time.time() - start_time
    print(f'i : {i+1 :4d}, {100*(i+1) / len(img_data_loader):.2f} % |   | Time: {time_pass//(60*60):3.0f}h, {time_pass//60%60 :.0f}m, {time_pass%60 :.0f}s.', end="\r")


# smoothes = []
# for i, (images, labels) in enumerate(img_data_loader):
#     images, labels  = images.to(device), labels.to(device)
#     smooth = []
#     for image, label in  zip(images, labels):
#         s = smooth_grad(image, label, model, args.epoch , args.sigma_multiplier)
#         smooth.append(s)
        
#     smooth = np.stack(smooth)
#     smoothes.append(smooth)
    
#     time_pass = time.time() - start_time
#     print(f'i : {i+1 :4d }, {100*(i+1) / len(img_data_loade):.1f} %, Data Size :{len(smoothes) :5d}  | Time: {time_pass/60%60 :.0f}m, {time_pass%60 :.0f}s.', end="\r")
# smoothes = np.stack(smoothes)

# smoothes = []

# start_time = time.time()

# smooth, smoothes = None, None

# # if args.start_iter > 0 :
# #     smoothes_ = np.load(f'{path_root}/EXAI_Visualization/smoothes_resize.npz')["x"]
# #     smoothes_

# for i, (images, labels) in enumerate(img_data_loader):

#     images, labels  = images.to(device), labels.to(device)

#     for j, (image, label) in  enumerate(zip(images, labels)):
#         smooth_ = smooth_grad(image, label, model, args.epoch , args.sigma_multiplier)
#         if j == 0 : 
#             smooth = smooth_
#         else:
#             smooth = np.concatenate((smooth, smooth_), axis=0)
            
#         time_pass = time.time() - start_time
        
#         if i==0:
#             print(f'i : {i+1 :4d}_{j+1:2d}, {100*(i+1) / len(img_data_loader):.2f} % | Data Size :{smooth.shape}  | Time: {time_pass//(60*60):3.0f}h, {time_pass//60%60 :.0f}m, {time_pass%60 :.0f}s.', end="\r")
#         else:
#             print(f'i : {i+1 :4d}_{j+1:2d}, {100*(i+1) / len(img_data_loader):.2f} % | Data Size :{smooth.shape}, {smoothes.shape}  | Time: {time_pass//(60*60):3.0f}h, {time_pass//60%60 :.0f}m, {time_pass%60 :.0f}s.', end="\r")
        
    
#     smooth = np.array([skimage.transform.resize(np.transpose(s,(1,2,0)),  size) for s in smooth]).astype(np.float32)
    
#     if i ==0 :
#         smoothes = smooth
#     else:
#         smoothes = np.concatenate((smoothes, smooth), axis=0)

#     smoothes = np.concatenate((smoothes, smooth), axis=0)

    
#     if i%10 == 0 :
#         np.savez_compressed(f'{path_root}/EXAI_Visualization/smoothes_resize_{s}-{e}.npz', x=smoothes)
#         print("smoothes_resize.npz Saved")