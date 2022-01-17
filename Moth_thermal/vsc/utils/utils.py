from pathlib import Path
import torch
from torchvision.utils import make_grid, save_image
import torchvision.utils as vutils
import torchvision


def str_to_list(x): return [int(xi) for xi in x.split(',')]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg", ".png", ".jpeg", ".bmp"])


def record_scalar(writer, scalar_list, scalar_name_list, cur_iter):
    scalar_name_list = scalar_name_list[1:-1].split(',')
    for idx, item in enumerate(scalar_list):
        writer.add_scalar(scalar_name_list[idx].strip(' '), item, cur_iter)


# def record_image(writer, image_list, cur_iter):
#     image_to_show = torch.cat(image_list, dim=0)
#     writer.add_image('visualization', make_grid(
#         image_to_show, nrow=8), cur_iter)

def record_image(writer, image_list, cur_iter, nrow=8):
    image_to_show = torch.cat(image_list, dim=0)
    writer.add_image('visualization', make_grid(
        image_to_show, nrow=nrow), cur_iter)

# def load_model(model, pretrained):
#     weights = torch.load(pretrained)
#     pretrained_dict = weights['model'].state_dict()
#     model_dict = model.state_dict()
#     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#     model_dict.update(pretrained_dict)
#     model.load_state_dict(model_dict)


def load_model(model, pretrained, optims: str = None, map_location:str = None):
    weights = torch.load(pretrained, map_location=map_location)

    try:
        pretrained_model_dict = weights['model'].state_dict()
    except:
        pretrained_model_dict = weights
    model_dict = model.state_dict()
    pretrained_model_dict = {
        k: v for k, v in pretrained_model_dict.items() if k in model_dict}
    model_dict.update(pretrained_model_dict)
    model.load_state_dict(model_dict)

    if optims:
        pretrained_optims = weights['optims']
        assert(isinstance(pretrained_optims, list))
        for optim_idx, pretrained_optim in enumerate(pretrained_optims):
            pretrained_optim_dict = pretrained_optim.state_dict()
            optim_dict = optims[optim_idx].state_dict()
            pretrained_optim_dict = {
                k: v for k, v in pretrained_optim_dict.items() if k in optim_dict}
            optim_dict.update(pretrained_optim_dict)
            optims[optim_idx].load_state_dict(optim_dict)


def save_checkpoint(model, optims, epoch):
    model_out_path = f"pretrained/vsc_epoch_{epoch}.pth"
    state = {"epoch": epoch, "model": model, "optims": optims}
    torch.save(state, model_out_path)
    print(f"=========> Checkpoint saved to {model_out_path}")
    return model_out_path


def save_log(log_save_path, epoch, am_rec, am_prior, cur_iter: int = 0):
    with open(log_save_path, 'a') as loss_log:
        loss_log.write(
            ",".join([
                str(epoch),
                str(cur_iter),
                f'{am_rec.avg:.4f}',
                f'{am_prior.avg:.4f}',
                # f'{am_levelN.avg:.4f}',
                '\n'
            ])
        )


def log_rec_images(epoch: int, real: torch.tensor, rec: torch.tensor, fake: torch.tensor = None,
                   flag: str = 'train', resize: int = 128, save: bool = True, return_: bool = False,
                   nrow: int = 8, outf: Path = Path('results')):

    if flag == 'train':
        images_stack = torch.cat(
            [real[:2*nrow], rec[:2*nrow], fake[:2*nrow]], dim=0)
        save_path = outf.joinpath('vsc', f'image_epoch_{epoch:05d}.jpg')
        save_path_up = outf.joinpath(f'image_up_to_date.jpg')

    elif flag == 'valid':
        images_stack = torch.cat([real, rec], dim=0)
        save_path = outf.joinpath(
            'vsc_valid', f'benchmarks_epoch_{epoch:05d}.jpg')

    images_stack = torchvision.transforms.Resize(
        resize)(images_stack) if resize else images_stack

    if save:
        vutils.save_image(images_stack.data.cpu(), save_path, nrow=nrow)
        if flag == 'train':
            vutils.save_image(images_stack.data.cpu(),
                              save_path_up, nrow=nrow)
    if return_:
        return images_stack
