'''
This file is used to test the model output.
'''
import os
import torch
from torch import nn
from torch.autograd import Variable
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import utils

from generators import define_Gen
from discriminators import define_Dis


def test(args):

    transform = transforms.Compose(
        [transforms.Resize((args.crop_height,args.crop_width)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    dataset_dirs = utils.get_testdata_link(args.dataset_dir)

    a_test_data = datasets.ImageFolder(dataset_dirs['testA'], transform=transform)
    b_test_data = datasets.ImageFolder(dataset_dirs['testB'], transform=transform)

    a_test_loader = torch.utils.data.DataLoader(a_test_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    b_test_loader = torch.utils.data.DataLoader(b_test_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

    G_AtoB = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, norm=args.norm, use_dropout= not args.no_dropout, gpu_ids=args.gpu_ids)
    G_BtoA = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, norm=args.norm, use_dropout= not args.no_dropout, gpu_ids=args.gpu_ids)

    utils.print_networks([G_AtoB, G_BtoA], ['G_AtoB', 'G_BtoA'])

    try:
        ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
        G_AtoB.load_state_dict(ckpt['G_AtoB'])
        G_BtoA.load_state_dict(ckpt['G_BtoA'])
    except:
        print(' [*] No checkpoint! ')


    for i, (a_real_test, b_real_test) in enumerate(zip(a_test_loader, b_test_loader)):
        a_real_test = Variable(a_real_test[0], requires_grad=True)
        b_real_test = Variable(b_real_test[0], requires_grad=True)
        a_real_test, b_real_test = utils.cuda([a_real_test, b_real_test])
                
        G_AtoB.eval()
        G_BtoA.eval()

        with torch.no_grad():
            a_fake_test = G_BtoA(b_real_test)
            b_fake_test = G_AtoB(a_real_test)
            a_recon_test = G_BtoA(b_fake_test)
            b_recon_test = G_AtoB(a_fake_test)

        pic = (torch.cat([a_real_test, b_fake_test, a_recon_test, b_real_test, a_fake_test, b_recon_test], dim=0).data + 1) / 2.0

        if not os.path.isdir(args.results_dir):
            os.makedirs(args.results_dir)

        torchvision.utils.save_image(pic, args.results_dir+'/sample_' + str(i) + '.jpg', nrow=3)
