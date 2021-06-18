import os
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset

class WorldMothDataset(Dataset):
    def __init__(self, root, X, y=None, transforms=None):
        '''
        - 載入世界各地博物館蛾類影像資料集
        Args:
            image_root: path to image file
            X : 影像名稱(eg: 'xxx.jpg')
            y : 物種或類群名 
        '''
        self.root = root
        self.X = X
        self.y = y
        self.transforms = transforms
        
    def __getitem__(self, index):
        image_name = self.X
        image = Image.open(os.path.join(self.root, image_name[index]))
        if self.transforms is None:
            image = self.transforms.ToTensor(image)  # 轉為Tensor
        else:
            image = self.transforms(image)            # 使用指定的轉型方式
        
        if self.y is None:
            return image
        else:
            return image, self.y[index]
    def __len__(self):
        return len(self.X)
