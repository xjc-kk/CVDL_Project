'''
This file defines the cycleGAN model and training steps.
'''
import itertools
import functools

import os
import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import utils

from generators import define_Gen
from discriminators import define_Dis
from network import set_grad
from torch.optim import lr_scheduler


class cycleGAN(object):
    '''
    train is implemented as a member function
    '''
    def __init__(self,args):

        # Generators and Discriminators
        self.G_AtoB = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, norm=args.norm, use_dropout= not args.no_dropout, gpu_ids=args.gpu_ids)
        self.G_BtoA = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, norm=args.norm, use_dropout= not args.no_dropout, gpu_ids=args.gpu_ids)
        self.D_A = define_Dis(input_nc=3, ndf=args.ndf, norm=args.norm, gpu_ids=args.gpu_ids)
        self.D_B = define_Dis(input_nc=3, ndf=args.ndf, norm=args.norm, gpu_ids=args.gpu_ids)

        utils.print_networks([self.G_AtoB, self.G_BtoA, self.D_A, self.D_B], ['G_AtoB', 'G_BtoA', 'D_A', 'D_B'])

        # MSE loss and L1 loss
        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()

        # Optimizers and lr_scheduler
        self.g_optimizer = torch.optim.Adam(itertools.chain(self.G_AtoB.parameters(), self.G_BtoA.parameters()), lr=args.lr, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(itertools.chain(self.D_A.parameters(), self.D_B.parameters()), lr=args.lr, betas=(0.5, 0.999))
        
        self.g_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer, lr_lambda=utils.LambdaLR(args.epochs, 0, args.decay_epoch).step)
        self.d_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_optimizer, lr_lambda=utils.LambdaLR(args.epochs, 0, args.decay_epoch).step)

        # Check if there is a checkpoint
        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

        try:
            ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
            self.start_epoch = ckpt['epoch']
            self.D_A.load_state_dict(ckpt['D_A'])
            self.D_B.load_state_dict(ckpt['D_B'])
            self.G_AtoB.load_state_dict(ckpt['G_AtoB'])
            self.G_BtoA.load_state_dict(ckpt['G_BtoA'])
            self.d_optimizer.load_state_dict(ckpt['d_optimizer'])
            self.g_optimizer.load_state_dict(ckpt['g_optimizer'])
        except:
            print(' [*] No checkpoint! Train from the beginning! ')
            self.start_epoch = 0


    def train(self,args):
        # data transformation
        transform = transforms.Compose(
                                    [transforms.RandomHorizontalFlip(),
                                    transforms.Resize((args.load_height,args.load_width)),
                                    transforms.RandomCrop((args.crop_height,args.crop_width)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        dataset_dirs = utils.get_traindata_link(args.dataset_dir)

        # Dataloader for class A and B
        a_loader = torch.utils.data.DataLoader(datasets.ImageFolder(dataset_dirs['trainA'], transform=transform), 
                                                batch_size=args.batch_size, shuffle=True, num_workers=4)
        b_loader = torch.utils.data.DataLoader(datasets.ImageFolder(dataset_dirs['trainB'], transform=transform), 
                                                batch_size=args.batch_size, shuffle=True, num_workers=4)

        # get fake samples from the sample pool
        a_fake_sample = utils.Sample_from_Pool()
        b_fake_sample = utils.Sample_from_Pool()

        for epoch in range(self.start_epoch, args.epochs):

            lr = self.g_optimizer.param_groups[0]['lr']
            print('learning rate = %.7f' % lr)

            for i, (a_real, b_real) in enumerate(zip(a_loader, b_loader)):
                '''
                Generator First, Discriminator Second
                '''
                # Generator Optimization
                set_grad([self.D_A, self.D_B], False)
                self.g_optimizer.zero_grad()

                a_real = Variable(a_real[0])
                b_real = Variable(b_real[0])
                a_real, b_real = utils.cuda([a_real, b_real])

                a_fake = self.G_BtoA(b_real)
                b_fake = self.G_AtoB(a_real)

                a_recon = self.G_BtoA(b_fake)
                b_recon = self.G_AtoB(a_fake)

                a_idt = self.G_BtoA(a_real)
                b_idt = self.G_AtoB(b_real)

                # Identity losses
                a_idt_loss = self.L1(a_idt, a_real) * args.lamda * args.idt_coef
                b_idt_loss = self.L1(b_idt, b_real) * args.lamda * args.idt_coef

                # Adversarial losses
                a_fake_dis = self.D_A(a_fake)
                b_fake_dis = self.D_B(b_fake)

                a_real_label = utils.cuda(Variable(torch.ones(a_fake_dis.size())))
                b_real_label = utils.cuda(Variable(torch.ones(b_fake_dis.size())))

                a_gen_loss = self.MSE(a_fake_dis, a_real_label)
                b_gen_loss = self.MSE(b_fake_dis, b_real_label)

                # Cycle consistency losses
                a_cycle_loss = self.L1(a_recon, a_real) * args.lamda
                b_cycle_loss = self.L1(b_recon, b_real) * args.lamda

                # Total generators losses
                gen_loss = a_gen_loss + b_gen_loss + a_cycle_loss + b_cycle_loss + a_idt_loss + b_idt_loss

                # Update generators
                gen_loss.backward()
                self.g_optimizer.step()


                # Discriminator Optimization
                set_grad([self.D_A, self.D_B], True)
                self.d_optimizer.zero_grad()

                # Sample from history of generated images
                a_fake = Variable(torch.Tensor(a_fake_sample([a_fake.cpu().data.numpy()])[0]))
                b_fake = Variable(torch.Tensor(b_fake_sample([b_fake.cpu().data.numpy()])[0]))
                a_fake, b_fake = utils.cuda([a_fake, b_fake])

                a_real_dis = self.D_A(a_real)
                a_fake_dis = self.D_A(a_fake)
                b_real_dis = self.D_B(b_real)
                b_fake_dis = self.D_B(b_fake)

                a_real_label = utils.cuda(Variable(torch.ones(a_real_dis.size())))
                a_fake_label = utils.cuda(Variable(torch.zeros(a_fake_dis.size())))
                b_real_label = utils.cuda(Variable(torch.ones(b_real_dis.size())))
                b_fake_label = utils.cuda(Variable(torch.zeros(b_fake_dis.size())))

                # Discriminator losses
                a_dis_real_loss = self.MSE(a_real_dis, a_real_label)
                a_dis_fake_loss = self.MSE(a_fake_dis, a_fake_label)
                b_dis_real_loss = self.MSE(b_real_dis, b_real_label)
                b_dis_fake_loss = self.MSE(b_fake_dis, b_fake_label)

                # Total discriminators losses
                a_dis_loss = (a_dis_real_loss + a_dis_fake_loss)*0.5
                b_dis_loss = (b_dis_real_loss + b_dis_fake_loss)*0.5

                # Update discriminators
                a_dis_loss.backward()
                b_dis_loss.backward()
                self.d_optimizer.step()
                
                # print some information
                if (i + 1) % 20 == 0:
                    print("Epoch: (%3d) (%5d/%5d) | Gen Loss:%.2e | Dis Loss:%.2e" % 
                        (epoch, i + 1, min(len(a_loader), len(b_loader)), gen_loss, a_dis_loss + b_dis_loss))

            # Update the checkpoint
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'D_A': self.D_A.state_dict(),
                                   'D_B': self.D_B.state_dict(),
                                   'G_AtoB': self.G_AtoB.state_dict(),
                                   'G_BtoA': self.G_BtoA.state_dict(),
                                   'd_optimizer': self.d_optimizer.state_dict(),
                                   'g_optimizer': self.g_optimizer.state_dict()},
                                  '%s/latest.ckpt' % (args.checkpoint_dir))

            # Update learning rates
            self.g_lr_scheduler.step()
            self.d_lr_scheduler.step()
