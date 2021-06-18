# 載入class內會使用到的函式庫
import os
import pandas as pd
import numpy as np
import torchvision import transforms
from torch.utils.data.dataset import Dataset
from PIL import Image


# 直接從csv檔讀取資料
class MothDataset(Dataset):
    def __init__(self, image_root, csv_path,  transform=None):
        """
        1. 直接讀入csv檔
        2. 從csv檔中抓出標籤：海拔高度
        3. 從csv檔中直接抓出'影像名稱'，再用影像名稱開啟影像資料夾內的檔案
        Args:
            image_root: path to image file
            csv_path: path to csv file
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Get image list
        self.root   = image_root                              # 讀取影像資料夾 
        self.data   = pd.read_csv(csv_path , sep='\t')        # 讀取資料    
        # assign 'Alt' as 'Label'
        self.labels = np.asarray(self.data.loc[:, 'Alt'])     # 指定海拔作為label
        # get file_name
        self.names  = np.asarray(self.data.loc[:, 'image_name'])  # 抓出檔案名
        
        # Transforms
        self.to_tensor = transforms.ToTensor()
        self.transform = transform

    def __getitem__(self, index):
        label = self.labels[index]           # 讀取檔案標籤
        image_name =  self.names[index]      # 讀取檔名
        img_path = os.path.join(self.root, image_name)
        image = Image.open(img_path)         # 根據檔名開啟影像。得到image.jpg的物件　

        
        # 影像資料轉型
        if self.transform is not None:
            image = self.transform(image)    # 使用指定的轉型方式
        else:
            image = self.to_tensor(image)    # 轉為Tensor
        
        
        return image, label
    
    def __len__(self):
        # Calculate data size
        return len(self.data.index)
    
# csv檔先預處理過，需先切分後再送入
class MothImgDataset(Dataset):
    def __init__(self, image_root, X, y, transform=None):
        """
        1. 對應的csv檔應在送進資料前先整理為X與y的Series, Numpy 或List格式
        2. 資料送入前，應先切分好train, valid, test
        Args:
            image_root: path to image file
            X : 放置numpy或list格式資料，這裡為csv檔內的影像名稱('Number.jpg')
            y : 放置numpy或list格式標籤，這裡為csv檔內的海拔('Alt')
 
        """    
        self.root = image_root               # 讀取影像資料夾  
        # get file_name
        self.y = y
        self.labels = np.asarray(self.y)     # 指定海拔作為label
        # assign 'Alt' as 'Label'
        self.X = X
        self.names  = np.asarray(self.X)     # 抓出檔案名
        
        # Transforms
        self.transform = transform
        self.to_tensor = transforms.ToTensor()


    def __getitem__(self, index):
        label = self.labels[index]           # 讀取檔案標籤
        
        image_name =  self.names[index]      # 根據檔名開啟影像。得到image.jpg的物件
        img_path = os.path.join(self.root, image_name)
        image = Image.open(img_path)
        
        # 影像資料轉型
        if self.transform is not None:
            image = self.transform(image)    # 使用指定的轉型方式
        else:
            image = self.to_tensor(image)    # 轉為Tensor
        
        return image, label
    
    def __len__(self):
        # Calculate data size
        return len(self.X)


# byYY
class MothImageDataset_YY(Dataset):
    def __init__(self, root, image_name, alt=None, augmentation=False):
        # --------------------------------------------
        # Initialize paths, transforms, and so on
        # --------------------------------------------
        self.root = root
        self.image_name = image_name
        self.alt = alt
        
        if augmentation:
            self.transform = transforms.Compose([transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.2), ratio=(1, 1)),  
                                                 transforms.ColorJitter(), 
                                                 transforms.RandomRotation(10),
                                                 transforms.RandomHorizontalFlip(), 
                                                 transforms.ToTensor(), 
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                ])
        else:
            self.transform = transforms.Compose([transforms.Resize((224, 224)), 
                                                 transforms.ToTensor(), 
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                ])
        
    def __getitem__(self, index):
        # --------------------------------------------
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        # --------------------------------------------
        image = Image.open(os.path.join(self.root, self.image_name[index]))
        if self.transform is not None:
            image = self.transform(image)
        
        if self.alt is None:
            return image
        else:
            return image, self.alt[index]
        
    def __len__(self):
        # --------------------------------------------
        # Indicate the total size of the dataset
        # --------------------------------------------
        return len(self.image_name)