import os, time 
import torch

"""模型的存取"""

# 檢查資料夾是否已建立
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
    print('checkpoint folder made!')
    
def save_checkpoint(model=None, optimizer=None, path=None, epoch=None, best_loss=None):
    '''儲存模型
    model:      指定模型
    optimizer:  指定優化器
    path:       模型儲存的路徑
    epoch:      預設為None
    best_loss:  儲存記錄到的最佳loss值
    '''
    state = {'model_state_dict'    : model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'epoch':epoch,
             'best_loss':best_loss}
    torch.save(state , path)

def load_checkpoint(model, optimizer, path, model_name=None):
    '''載入中繼的模型
    model:      指定模型
    optimizer:  指定優化器
    path:       模型儲存的路徑
    '''
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_ = checkpoint['epoch']
    best_loss_ = checkpoint['best_loss']
    print(f'best_loss:{best_loss_:.4f}')
    if model_name==None:
        print('請輸入".pth"檔名')
    else:
        print(f'{model_name}.pth loaded!')
    return optimizer