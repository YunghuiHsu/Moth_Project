import os, time 
import torch
import pandas as pd
import numpy as np
from torch import  nn
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

# criterion = nn.CrossEntropyLoss() #  多項分類模型
criterion = nn.MSELoss()            #  迴歸模型

# 檢查變數是否已命名
try:
    train_log, valid_log, lr_log
    print('log file has bean setted')
except NameError:
    train_log , valid_log, lr_log = [], [], []
    print('log file resetted:\n    train_log, valid_log, lr_log')
#     train_Acc_log, valid_Acc_log   = [], []
    

# 檢查資料夾是否已建立
if not os.path.isdir('log'):
    os.mkdir('log')
    print('"log" folder made!')
if not os.path.isdir('figure'):
    os.mkdir('figure') 
    print('"figure" folder made!')

# save Loss log
def save_log(fileName=None):
    '''儲存學習過程'''
    (pd.DataFrame({"train_Loss":train_log   , 'valid_Loss':valid_log, 
#                    'train_Acc':train_Acc_log, 'valid_Acc' :valid_Acc_log,
                   'lr':lr_log})
       .to_csv(f'./log/Loss_log_{fileName}.csv', index=False))

# 分類模型時使用
def __getAccuracy(pred, labels, correct, total, get_feature=False):
    '''分類模型中計算分類正確率，在train、evaluate函式中使用'''
    pred_ = torch.argmax(pred, dim=1)              # 取得最大值的索引位置(0~9)
    correct += (pred_ == labels).sum().item()      # 累加預測正確的樣本數
    total += labels.size(0)                        # 累加每批次樣本數
    return (100*correct / (total+1e-10))           # 回傳分類正確率

def train(model, optimizer, dataloader, device, get_feature=False):
    '''訓練模型'''
    running_loss = 0.0         # 每輪歸零重計
#     correct ,total = 0.0, 0.0  
    model.train()  # 明確指定model在訓練狀態(預設值)
    for i, (inputs, labels) in enumerate(dataloader): 
        inputs, labels = inputs.to(device), labels.to(device)  
        optimizer.zero_grad()                          # 將優化器梯度歸零 
        
        if get_feature:
            _, pred = model(inputs)                    # model()會回傳2個值，僅接收預測值(注意model定義中的回傳值)
        else:
            pred = model(inputs)
            
        loss = criterion(pred, labels.view(-1,1))      # 計算LOSS ， 調整為形狀一致
        loss.backward()                                # 反向傳導
        optimizer.step()                               # 更新參數
        running_loss += loss.item()                    # 累加這輪epoch的loss 
        # get accuracy
#         acc = __getAccuracy(pred, labels, correct, total)     
    # 每輪epoch結束後取平均loss
    mean_loss = running_loss/i
    train_log.append(mean_loss)
    lr_log.append(optimizer.param_groups[0]['lr']) 
    return mean_loss
#     return mean_loss , acc                             # 回傳平均loss, 分類正確率
#     train_Acc_log.append(acc)                        # 計算分類正確率 

@torch.no_grad()
def evaluate(model, dataloader, device, get_feature=False):
    '''進行推論評估模型在驗證資料集的誤差'''
    running_loss = 0.0         # 每輪歸零重計
#     correct ,total = 0.0, 0.0    
    model.eval() # 啟動評估模式
    for i, (inputs, labels) in enumerate(dataloader):  
        inputs, labels = inputs.to(device), labels.to(device)
        if get_feature:
            _, pred = model(inputs)                     
        else:
            pred = model(inputs)
        loss = criterion(pred, labels.view(-1,1))
        running_loss += loss.item()
#         acc = __getAccuracy(pred, labels, correct, total) 
    # 每輪epoch結束後取平均loss
    mean_loss = running_loss/i
    valid_log.append(mean_loss)
    return mean_loss
#     valid_Acc_log.append(acc)
#     return mean_loss , acc


@torch.no_grad()                    
def test(model, dataloader ,device , save=False, model_name=None):
    '''
    從測試資料集取得所有測試及標籤與預測結果，在moth alt資料集中還包括取出特徵值(feature)
    上面一行加上 @torch.no_grad() 指定不進行梯度計算
    model:      放入指定的模型
    dataloader: 預設使用測試資料集
    device:     指定cpu或gpu
    save:       是否存檔
    '''
    model.eval()
    for i, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        feature, outputs = model(images)
        
        # 取出
        if i == 0:   # 設定初始值
            fea_   =  feature.detach().detach().cpu().numpy() 
            y_test =  labels.detach().detach().cpu().numpy()
            y_pre  =  outputs.detach().detach().cpu().numpy()
        else:   
            fea_   = np.concatenate((fea_,   feature.detach().cpu().numpy())) 
            y_test = np.concatenate((y_test,  labels.detach().cpu().numpy())) 
            y_pre  = np.concatenate((y_pre,  outputs.detach().cpu().numpy()))
    if save:
        np.save(f'./meta/feature_{model_name}.npy', fea_)
        np.save(f'./meta/y_test_{model_name}.npy' , y_test)
        np.save(f'./meta/y_pre_{model_name}.npy'  , y_pre)
    return y_test, y_pre

def get_LossFig(figName=None, s=0, e=None , savefig=False):
    '''繪製學習過程圖
    Parameters
     ----------
       figName: str, optional, default: None
       s: scalar, optional, default: 0
          start of epoch.  number(int), optional
       e: scalar, optional, default: None
          end of epoch.    
       savefig: boolean
           whether to save
   '''
    log = pd.read_csv(f'./log/Loss_log_{figName}.csv')
    log = log[s:e] 
    fig = log[['train_Loss','valid_Loss']].plot( grid=True, lw=3)
    ax2 = fig.twinx()
    ax2.plot(np.log10(log.lr) , label='lr', c='g', ls=':', lw=3)   # lr取log 
    ax2.legend(['LR'],  loc='upper center')
    plt.title('LOSS', {'fontsize':16})
    plt.xlabel('Epoch'), fig.set_ylabel('Loss_MSE',{'fontsize':14}), ax2.set_ylabel('lr_rate(log)',{'fontsize':14})

    # 圖片存檔
    if savefig:       
        plt.savefig(f'./figure/Loss_{figName}.png', bbox_inches='tight')
    fig;
    
def early_stop(valid_loss, best_loss,  trigger_times, patience):
    '''早停機制
    patience:　啟動後。等待幾回合
    '''
    if valid_loss > best_loss:
        trigger_times += 1
        print('  Early stopping trigger times:', trigger_times)
    else:
        trigger_times = 0

    return trigger_times
