import torch
from model import NetG, NetD
import cv2
import numpy as np
import torchvision.utils as vutils
from PIL import Image
import os

def test(opt):
    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    PATH_G = 'pth/netG_025.pth'
    netG = NetG(opt.ngf, opt.nz)
    netG.load_state_dict(torch.load(PATH_G))
    netG.to(device)

    PATH_D = 'pth/netD_025.pth'
    netD = NetD(opt.ndf)
    netD.load_state_dict(torch.load(PATH_D))
    netD.to(device)


    # model eval to Test pattern
    netG.eval()
    # generate 256 noises to select the best 32 fake imgs
    noise = torch.randn(256, opt.nz, 1, 1)
    noise = noise.to(device)
    result = []
    with torch.no_grad():
        output_img = netG(noise)
        scores = netD(output_img).squeeze()
        _, indexs = scores.topk(32) # [0]: content, [1] : index
        for idx in indexs:
            result.append(output_img.data[idx])
    imgs = torch.stack(result)

    if not os.path.exists(opt.test_path):
        os.makedirs(opt.test_path)
    vutils.save_image(imgs, "%s/test.png" % opt.test_path, normalize=True)

    # normalize image
    # def norm_ip(img, min, max):
    #     img.clamp_(min=min, max=max)
    #     img.add_(-min).div_(max-min + 1e-5)
    #     return img
    # output_img = norm_ip(output_img, float(output_img.min()), float(output_img.max()))
    # img_arr = output_img.squeeze().mul(255).add_(0.5).clamp_(0, 255).permute(1,2,0).to('cpu', torch.uint8).numpy()
    # cv2.imwrite("%s/test.png" % opt.test_path, cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR))










