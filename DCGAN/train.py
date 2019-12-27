import torch
import torchvision
import torchvision.utils as vutils
import torch.nn as nn
from random import randint
from model import NetD, NetG
import os

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(opt):
    # Image Input
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(opt.imageSize),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = torchvision.datasets.ImageFolder(opt.data_path, transform=transforms)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        drop_last=True
    )

    netG = NetG(opt.ngf, opt.nz).to(device)
    netD = NetD(opt.ndf).to(device)
    if torch.cuda.device_count() > 1:
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)

    criterion = nn.BCELoss()
    optimizeG = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizeD = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    label = torch.FloatTensor(opt.batchSize)
    real_label = 1
    fake_label = 0

    for epoch in range(1, opt.epoch+1):
        for i, (imgs, _) in enumerate(dataloader):
            # fix Generator, update Discriminator
            optimizeD.zero_grad()
            # real imgs
            imgs = imgs.to(device)
            output = netD(imgs)
            label.data.fill_(real_label)
            label = label.to(device)
            errD_real = criterion(output.squeeze(), label)
            errD_real.backward()
            # fake imgs
            noise = torch.randn(opt.batchSize, opt.nz, 1, 1)
            noise = noise.to(device)
            fake_imgs = netG(noise)
            output = netD(fake_imgs.detach()) # Avoid the gradients delivery to G
            label.data.fill_(fake_label)
            errD_fake = criterion(output.squeeze(), label)
            errD_fake.backward()
            errD = errD_fake + errD_real
            optimizeD.step()

            # fix Discriminator, update Generator
            optimizeG.zero_grad()
            # As far as possible to fake Disciminator by GNet()
            output = netD(fake_imgs)
            label.data.fill_(real_label)
            label.to(device)
            errG = criterion(output.squeeze(), label)
            errG.backward()
            optimizeG.step()

            print('[%d/%d][%d/%d] Loss_D: %.3f Loss_G %.3f'
                % (epoch, opt.epoch, i, len(dataloader), errD.item(), errG.item()))
        if not os.path.exists(opt.outf):
            os.makedirs(opt.outf)
        vutils.save_image(fake_imgs.data, '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch), normalize=True)
        if not os.path.exists(opt.pth):
            os.makedirs(opt.pth)
        torch.save(netG.state_dict(), '%s/netG_%03d.pth' % (opt.pth, epoch))
        torch.save(netD.state_dict(), '%s/netD_%03d.pth' % (opt.pth, epoch))
