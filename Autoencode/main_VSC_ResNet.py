# python main_vsc.py 
from __future__ import print_function
import os, time, random, argparse, glob, re
from math import log10
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
# from tensorboardX import SummaryWriter

from MothScripts.dataset import ImageDatasetFromFile
from MothScripts.networks import DFC_VSC_WGAN
from MothScripts.resnet50_classifier import ResNet50_family_classifier
from MothScripts.average_meter import AverageMeter
#import visdom
#viz = visdom.Visdom()

if not os.path.exists("model/vsc_wgan/"):
    os.makedirs("model/vsc_wgan/")

parser = argparse.ArgumentParser()
parser.add_argument('--channels', default="32, 64, 128, 256, 512, 512", type=str, help='the list of channel numbers')
parser.add_argument("--hdim", type=int, default=512, help="dim of the latent code, Default=512")
parser.add_argument("--save_iter", type=int, default=1, help="Default=1")
parser.add_argument("--test_iter", type=int, default=2000, help="Default=1000")
parser.add_argument('--nrow', type=int, help='the number of images in each row', default=8)
parser.add_argument('--dataroot', 
                    default="/home/jovyan/Autoencoder/wolrdwide_lepidoptera_yolov4_cropped_and_padded_20210610", 
                    type=str, help='path to dataset')
parser.add_argument('--trainsize', type=int, help='number of training data', default=-1)
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--input_height', type=int, default=256, help='the height  of the input image to network')
parser.add_argument('--input_width', type=int, default=256, help='the width  of the input image to network')
parser.add_argument('--output_height', type=int, default=256, help='the height  of the output image to network')
parser.add_argument('--output_width', type=int, default=256, help='the width  of the output image to network')
parser.add_argument("--nEpochs", type=int, default=15000, help="number of epochs to train for")
parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
# parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--parallel', action='store_true', help='for multiple GPUs')
parser.add_argument('--outf', default='results/vsc_wgan/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--tensorboard', action='store_true', help='enables tensorboard')
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")

str_to_list = lambda x: [int(xi) for xi in x.split(',')]

def is_image_file(filename):
    return any(filename.lower().endswith(extension) for extension in [".jpg", ".png", ".jpeg",".bmp"])
    
def record_scalar(writer, scalar_list, scalar_name_list, cur_iter):
    scalar_name_list = scalar_name_list[1:-1].split(',')
    for idx, item in enumerate(scalar_list):
        writer.add_scalar(scalar_name_list[idx].strip(' '), item, cur_iter)

def record_image(writer, image_list, cur_iter):
    image_to_show = torch.cat(image_list, dim=0)
    writer.add_image('visualization', make_grid(image_to_show, nrow=opt.nrow), cur_iter)
    
def load_model(model, pretrained):
    weights = torch.load(pretrained)
    pretrained_dict = weights['model'].state_dict()  
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)
            
def save_checkpoint(model, epoch, iteration, prefix=""):
    model_out_path = "model/vsc_wgan/" + prefix +"model_local_epoch_{}_iter_{}.pth".format(epoch, iteration)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("model/vsc_wgan/"):
        os.makedirs("model/vsc_wgan/")
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

    #--------------optional def -------------------------
    
def save_benchmark_imgs(model, epoch, benchmarks_data_loader):
    with torch.no_grad():
        for real_bench, denoised_bench, _ in benchmarks_data_loader:
            if opt.parallel:
                x = real_bench.cuda(1)
            else:
                x = real_bench.cuda()
            mu_, logvar_, logspike_ = model.encode(x)
            z_bench = model.reparameterize(mu_, logvar_, logspike_)
            rec_bench = model.decode(z_bench).cpu()
        vutils.save_image(
            torch.cat([denoised_bench , rec_bench], dim=0).data.cpu(),
            '{}/BMimage_Epoch{}.jpg'.format(opt.outf, epoch),
            nrow=10
        )
        print(f'\tBMimage_Epoch{epoch}.jpg saved')
    
def clear_checkpoint():
    ''' 
        delete the oldest checkpoint in  ./model/vsc_wgan/ 
    '''
    path = "/home/jovyan/Autoencoder/model/vsc_wgan/"
    find_models = os.listdir(path)
    find_models.sort()
    model_delete = re.compile(r'model_local_epoch_[0-9]{4,}_iter_0.pth')
    delete_list = model_delete.findall(
        (',').join(find_models)
    )
    model_delete = path + delete_list[0]         # 取出第一項，每次刪除一筆 
    cmd = f"rm -v  {model_delete}"              # 執行刪除
    os.system(cmd)

def clear_results_imgs():
    path = "/home/jovyan/Autoencoder/results/vsc_wgan/"
    cmd = f'rm -v {path}*.jpg'  # 刪除 ./results/vsc_wgan/內所有的jpg檔案
    os.system(cmd)    

def backup_results_imgs():
    '''
        backup images  in  ./results/vsc_wgan/
    '''
    path = "/home/jovyan/Autoencoder/results/vsc_wgan/"
    results_Imgs = os.listdir(path)
    # 先備份 ./results/vsc_wgan/內符合條件的pg檔案
    bm_backup = 'BMimage_Epoch[0-9]{1}[05]{1}[0]{2}.jpg'
    img_backup = 'image_[0-9]+[05]{1}[0]{4,}.jpg'
    for condition in [bm_backup, img_backup]:
        backup = re.compile(condition)
        backup_list = backup.findall(
            (',').join(results_Imgs)
        )
        if len(backup_list) > 0 :  # 當有找到符合條件的檔案時，強制執行備份至 results/vsc_wgan/backup/ 資料夾
            cmd_backup = f"cp -fv {path}{backup_list[0]} {path}'/backup/'{backup_list[0]}"              
            os.system(cmd_backup)

def main():
    
    global opt, model
    opt = parser.parse_args()
#     opt = parser.parse_args(args=[]) # in jupyter 
    print(opt)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)

    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    #if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

    cudnn.benchmark = True

    #if torch.cuda.is_available() and not opt.cuda:
    #    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    is_scale_back = False
    
    #--------------build models -------------------------

    
    if opt.parallel:
        model = DFC_VSC_WGAN(cdim=3, hdim=opt.hdim, channels=str_to_list(opt.channels), image_size=opt.output_height, parallel=True).cuda()
        resnet50 = ResNet50_family_classifier(parallel=True).cuda()
    else:
        model = DFC_VSC_WGAN(cdim=3, hdim=opt.hdim, channels=str_to_list(opt.channels), image_size=opt.output_height).cuda()
        resnet50 = ResNet50_family_classifier().cuda()

    # 載入pretrained的 ResNet50
    resnet_pretrained = torch.load("model/vsc_wgan/pretrained_fam_classification_resnet50_20210613.pth")
#     resnet_pretrained = torch.load("model/vsc_wgan/pretrained_fam_classification_resnet50_20210613.pth", map_location=lambda storage, loc: storage.cpu())
    pretrained_dict, resnet50_dict = resnet_pretrained['model_state'], resnet50.state_dict()      # 載入權重參數
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in resnet50_dict}            # 更新pretrained model的key與value，使之與resnet50一致
    resnet50_dict.update(pretrained_dict)                                                         # 將resnet50的權重更新為pretrained model
    resnet50.load_state_dict(resnet50_dict)                                                       # 將pretrained的權重載入resnet50
    resnet50.eval()

    pretrained_default = 'model/vsc_wgan/model_local_epoch_%d_iter_0.pth' % opt.start_epoch
    #pretrained_default = 'model/vsc/model_local_epoch_%d_iter_0.pth' % 2000

    if opt.pretrained:
        load_model(model, opt.pretrained)
    elif os.path.isfile(pretrained_default):
        print ("Loading default pretrained %d..." % opt.start_epoch)
        load_model(model, pretrained_default)

    #print(model)

#     optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    #-----------------load dataset--------------------------
    image_list = [x for x in glob.iglob(opt.dataroot + '/**/*', recursive=True) if is_image_file(x)]
    #train_list = image_list[:opt.trainsize]
    train_list = image_list[:]
    assert len(train_list) > 0
    
    train_set = ImageDatasetFromFile(train_list, aug=True)
    train_data_loader = DataLoader(train_set, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers), pin_memory=True)
    num_of_batches = len(train_data_loader)
    
    #載入評估用的benchmarks影像檔案(背景及花紋較複雜的指標照)
    benchmarks_list = [x for x in glob.iglob("./benchmarks" + '/**/*', recursive=True)]
    benchmarks_set = ImageDatasetFromFile(benchmarks_list, aug=True)
    benchmarks_data_loader = DataLoader(benchmarks_set, batch_size=30, num_workers=1, pin_memory=True)
    
    start_time = time.time()

    #cur_iter = 0        
    #cur_iter = int(np.ceil(float(opt.trainsize) / float(opt.batchSize)) * opt.start_epoch)
    cur_iter = len(train_data_loader) * (opt.start_epoch - 1)
    
    def train_vsc_wgan(epoch, iteration, batch, denoised_batch, cur_iter):
        
        #if len(batch.size()) == 3:
        #    batch = batch.unsqueeze(0)
        #print(batch.size())
            
        batch_size = batch.size(0)
                       
        real= Variable(batch).cuda() 
        denoised = Variable(denoised_batch).cuda()

        noise = Variable(torch.zeros(batch_size, opt.hdim).normal_(0, 1)).cuda()
        fake = model.sample(noise)

        time_cost = time.time()-start_time
        info = "=> Cur_iter: [{}]: Epoch[{}]({}/{}). Time:{:3.0f}h{:2.0f}m{:2.0f}s. ".format(
            cur_iter, 
            epoch, 
            iteration, 
            len(train_data_loader), 
            time_cost//(60*60), 
            time_cost//60%60, 
            time_cost%60
        )

        loss_info = '[loss_rec, loss_margin, lossE_real_kl, lossE_rec_kl, lossE_fake_kl, lossG_rec_kl, lossG_fake_kl,]'

        #=========== Update optimizer ================                  
        real_mu, real_logvar, real_logspike, z, rec, f_targ, f_pred, y = model(real, x_denoised=denoised, discriminator_model=resnet50)

        loss_Rec =  model.reconstruction_loss(f_targ, f_pred, size_average=True)
#         loss_KL = model.kl_loss(real_mu, real_logvar).mean()
        loss_Prior = model.prior_loss(real_mu, real_logvar, real_logspike)
        loss_W = model.W_loss(y)

        loss = loss_Rec + loss_Prior + loss_W*100

        optimizer.zero_grad()    
        loss.backward()                   
        optimizer.step()

        #info += 'Rec: {:.4f}, KL: {:.4f}, '.format(loss_rec.data[0], loss_kl.data[0])
        am_Loss.update(loss.item())
        am_RecLoss.update(loss_Rec.item())
        am_PriorLoss.update(loss_Prior.item())
#         am_KLLoss.update(loss_KL.item())
        am_WLoss.update(loss_W.item())

        info += '| Loss: {:,.0f}({:,.0f}), Rec: {:,.0f}({:,.0f}), Prior: {:.2f}({:.2f}), W: {:.2f}({:.2f})   '.format(
            am_Loss.val     , am_Loss.avg, 
            am_RecLoss.val  , am_RecLoss.avg,
            am_PriorLoss.val, am_PriorLoss.avg,
#             am_KLLoss.val   , am_KLLoss.avg,
            am_WLoss.val    , am_WLoss.avg
        )
#         , KL: {:.3f}({:.3f}), W: {:.4f}({:.4f}) 
        print(info, end='\r')
        
        # viz_idx = cur_iter
        # if cur_iter % 10 is 0:
        #     viz.line([[am_rec.avgN, am_rec.avg]], [viz_idx], win='loss_rec', update='append', opts=dict(title='Rec'))
        #     viz.line([[am_prior.avgN, am_prior.avg]], [viz_idx], win='loss_prior', update='append', opts=dict(title='Prior'))

        if cur_iter % opt.test_iter is 0:
            if opt.tensorboard:
                record_scalar(writer, eval(loss_info), loss_info, cur_iter)
                if cur_iter % 1000 == 0:
                    record_image(writer, [real, rec, fake], cur_iter)   
            else:
                vutils.save_image(
                    torch.cat([real[:16], rec[:16], fake[:16]], dim=0).data.cpu(),
                    '{}/image_{}.jpg'.format(opt.outf, cur_iter),
                    nrow=opt.nrow
                )
                #not_enough = rec - denoised
                #too_much = -not_enough
                #vutils.save_image(torch.cat([not_enough, too_much], dim=0).data.cpu(), '{}/overlay/image_{}.jpg'.format(opt.outf, cur_iter),nrow=opt.nrow)
                with open('./model/vsc_wgan/losses.log','a') as loss_log:
                    loss_log.write(
                        ",".join([
                            str(cur_iter),
                            str(epoch),
#                             '%.0f' % time_cost,
                            '%.0f' % am_Loss.avg,
                            '%.0f' % am_RecLoss.avg,
                            '%.4f' % am_PriorLoss.avg,
#                             '%.4f\n' % am_KLLoss.avg,
                            '%.4f\n' % am_WLoss.avg]))
        elif cur_iter % 1000 is 0:
            vutils.save_image(
                torch.cat([real[:8], rec[:8], fake[:8]], dim=0).data.cpu(),
                '{}/image_up_to_date.jpg'.format(opt.outf),
                nrow=opt.nrow
            )
    #----------------Train by epochs--------------------------
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):  
        model.train()
        model.c = 50 + epoch * model.c_delta

        am_Loss, am_RecLoss, am_PriorLoss, am_WLoss = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

        for iteration, (batch, denoised_batch, filenames) in enumerate(train_data_loader, 0):
            #--------------train------------
            train_vsc_wgan(epoch, iteration, batch, denoised_batch, cur_iter)            
            cur_iter += 1

        print(f'epoch : {epoch}, {len(train_data_loader)}')
        
        # save checkpoint and benchmarks
        save_epoch = (epoch//opt.save_iter)*opt.save_iter
        if epoch == save_epoch:
            save_checkpoint(model, save_epoch, 0, '')
            
            # optionnal operation
            clear_checkpoint()          
            backup_results_imgs()
            clear_results_imgs()
            save_benchmark_imgs(model, epoch, benchmarks_data_loader)
            

if __name__ == "__main__":
    main()    
