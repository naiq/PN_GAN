import torch
import torch.nn as nn
from torchvision import transforms, utils
import torch.utils.data as Data
from torch.autograd import Variable
from torch.optim import lr_scheduler
from config import cfg
from tensorboardX import SummaryWriter
import os, itertools
import network
import dataset
import time
import matplotlib.pyplot as plt


# Save images
def save_images(name, PATH, fake_img):
    samples = fake_img.data
    if not os.path.exists(PATH):
        os.mkdir(PATH)
    utils.save_image(samples, '%s/%s.png' % (PATH, name), nrow=1, padding=0, normalize=True)


# Load Data
def load_data(samples):
    val_data = dataset.Market_test(imgs_name=samples, imgs_path=cfg.TEST.imgs_path, pose_path=cfg.TEST.pose_path,
                                         transform=dataset.val_transform(), loader=dataset.val_loader)
    val_loader = Data.DataLoader(val_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                                 num_workers=cfg.TEST.BATCH_SIZE)

    val = [val_data, val_loader]
    return val


# Load Network
def load_network(model_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.TEST.GPU_ID
    print ('###################################')
    print ("#####      Load Network      #####")
    print ('###################################')

    nets = []
    netG = network.Res_Generator(cfg.TRAIN.ngf, cfg.TRAIN.num_resblock)
    netG.load_state_dict(torch.load(model_path)['state_dict'])

    nets.append(netG)
    for net in nets:
        net.cuda()

    print ('Finished !')
    return nets



def test(val_file, nets, save_path):
    print ('\n###################################')
    print ("#####      Start Testing      #####")
    print ('###################################')

    _, val_loader = val_file
    netG = nets[0]

    for _, (src_img, pose, name) in enumerate(val_loader):
        # #######################################################
        # (1) Data process
        # #######################################################
        src_img = Variable(src_img, volatile=True).cuda()      # N x 3 x H x W
        pose = Variable(pose, volatile=True).cuda()            # N x 3 x H x W


        # #######################################################
        # (2) Generate images
        # #######################################################
        fake_img = netG(src_img, pose)


        # #######################################################
        # (3) Save images
        # #######################################################
        save_images(name[0], save_path, fake_img)
        print ('Generate image: ', name[0])



def main(samples, model_path, save_path):
    val_file = load_data(samples)
    nets = load_network(model_path)
    test(val_file, nets, save_path)


if __name__ == '__main__':
    samples = ['0005_c3s3_060878_01.jpg']   # sample image
    model_path = 'model/YOUR_SAVED_MODEL'
    save_path = 'test'
    main(samples, model_path, save_path)

