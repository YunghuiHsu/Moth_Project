import torch
from torchvision.utils import make_grid, save_image

str_to_list = lambda x: [int(xi) for xi in x.split(',')]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg", ".png", ".jpeg",".bmp"])
    
def record_scalar(writer, scalar_list, scalar_name_list, cur_iter):
    scalar_name_list = scalar_name_list[1:-1].split(',')
    for idx, item in enumerate(scalar_list):
        writer.add_scalar(scalar_name_list[idx].strip(' '), item, cur_iter)

def record_image(writer, image_list, cur_iter):
    image_to_show = torch.cat(image_list, dim=0)
    writer.add_image('visualization', make_grid(image_to_show, nrow=8), cur_iter)

def load_model(model, pretrained):
    weights = torch.load(pretrained)
    pretrained_dict = weights['model'].state_dict()  
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)
