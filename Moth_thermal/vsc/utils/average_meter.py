import numpy as np
mid_thres = 460
class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self, margin_threshold=mid_thres + 80, std=False, moving_window_size=10):
        self.std = std
        self.reset()
        self.moving_window_size = moving_window_size
        self.cells = np.zeros(self.moving_window_size)

        self.margin_threshold_count = 0
        self.margin_threshold = margin_threshold

        self.thres = [mid_thres-60, mid_thres-40, mid_thres-20, mid_thres, mid_thres+20, mid_thres+40, mid_thres+60]

        self.margin0_count = 0
        self.margin1_count = 0
        self.margin2_count = 0
        self.margin3_count = 0
        self.margin4_count = 0
        self.margin5_count = 0
        self.margin6_count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.avgN = 0
        if self.std:
            self.all = []
        else:
            self.all = None

    def update_moving(self, val):
        self.cells[1:self.moving_window_size] = self.cells[0:(self.moving_window_size-1)]
        self.cells[0] = val
        self.avgN = np.mean(self.cells)

    def update(self, val, n=1):
        self.cells[1:self.moving_window_size] = self.cells[0:(self.moving_window_size-1)]
        self.cells[0] = val
        self.avgN = np.mean(self.cells[0:min(self.moving_window_size, self.count+1)])

        if self.std:
            self.all.append(val)
        
        self.val = val
        self.sum += val * n
        self.count += n
        thres = self.thres
        if val > self.margin_threshold:
            self.margin_threshold_count += n
        if val > thres[0]:
            self.margin0_count += n
        if val > thres[1]:
            self.margin1_count += n
        if val > thres[2]:
            self.margin2_count += n
        if val > thres[3]:
            self.margin3_count += n
        if val > thres[4]:
            self.margin4_count += n
        if val > thres[5]:
            self.margin5_count += n
        if val > thres[6]:
            self.margin6_count += n
            
        self.avg = self.sum / self.count
        self.over_threshold = self.margin_threshold_count / self.count

    def get_std(self): # no puns here
        if self.std:
            all = np.array(self.all)
            return np.std(all)
        return 0