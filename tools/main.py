import os
import torch
from lib.config.para import parse_opt
from lib.config.cfg import config as cfg
# use gpu
opt = parse_opt()
opt.CUDA_DEVICE = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.CUDA_DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import sys
import pprint
import pandas as pd
import os.path as osp
from collections import namedtuple

import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
from torch.utils.tensorboard import SummaryWriter

from lib.models.network import ResNet, Bottleneck
from lib.dataset.dataset_id import Intentonomy
from lib.utils.utils import read_ann_json, get_logger, ele_sum, update_center_by_dloss, get_center
from lib.core.function_id import train_prototype, evaluate_by_prototype
from lib.core.build_prototype import cluster_by_cosine,get_all_fea
from lib.criterion.diversity_loss import calculate_cls, Mydata, diversity_loss
from lib.evaluation.evaluation import get_allresults_df
from lib.prepare import Logger, prepare_env

import warnings
warnings.filterwarnings('ignore')


def main():
    cfg.BASIC.SEED = opt.seed = 50
    opt.epoch = 40
    opt.found_lr = 0.001
    opt.lr_prototype = 0.001
    opt.milestones = [20, 30]

    # create checkpoint directory
    cfg.BASIC.ROOT_DIR = osp.join(osp.dirname(__file__), '..')
    cfg.BASIC.SAVE_DIR = osp.join(cfg.BASIC.ROOT_DIR, 'ckpt',
                                  'sgd-adam_seed{}_epoch{}_steps{}_found-lr{}_prototype-lr{}_{}'.format(
                                opt.seed, opt.epoch, opt.milestones, opt.found_lr, opt.lr_prototype, cfg.BASIC.TIME))
    # prepare running environment for the whole project
    prepare_env(cfg)

    # start loging
    sys.stdout = Logger(cfg.BASIC.LOG_FILE)
    pprint.pprint(opt)
    pprint.pprint(cfg)
    writer = SummaryWriter(cfg.BASIC.LOG_DIR)

    # dataset
    train_anns = read_ann_json(opt.train_json_file)
    train_fun = Intentonomy(opt.img_dir, train_anns, phase='train')
    train_loader = torch.utils.data.DataLoader(train_fun, batch_size=opt.batch_size, shuffle=True, num_workers=cfg.BASIC.NUM_WORKERS)
    val_anns = read_ann_json(opt.val_json_file)
    val_fun = Intentonomy(opt.img_dir, val_anns, phase='val')
    val_loader = torch.utils.data.DataLoader(val_fun, batch_size=opt.batch_size, shuffle=False, num_workers=cfg.BASIC.NUM_WORKERS)
    test_anns = read_ann_json(opt.test_json_file)
    test_fun = Intentonomy(opt.img_dir, test_anns, phase='test')
    test_loader = torch.utils.data.DataLoader(test_fun, batch_size=opt.batch_size, shuffle=False, num_workers=cfg.BASIC.NUM_WORKERS)

    # network
    pretrained_model = models.resnet50(pretrained=True)
    ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])
    resnet50_config = ResNetConfig(block=Bottleneck,
                                   n_blocks=[3, 4, 6, 3],
                                   channels=[64, 128, 256, 512])
    model = ResNet(resnet50_config)
    model_static = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_model.state_dict().items() if k in model_static}
    model_static.update(pretrained_dict)
    model.load_state_dict(model_static)
    print('load network')
    model.to(device=device)

    # build
    print("strat build prototype")
    # prepare for prototype
    lab_and_pre = {}  # result of test and val
    max_score = 0  # result of max score in all epoch
    max_df = pd.DataFrame()
    temp_prototype = []
    prototype = []
    cn = []
    fea_all = []
    for i in range(28):
        t = list()
        fea_all.append(t)
    fea_all = get_all_fea(model, train_loader, fea_all)
    for data in fea_all:
        centers, cluster_number = cluster_by_cosine(data, opt.component)
        cn.append(cluster_number)
        prototype.append(centers)

    # save path
    best_score_save = osp.join(cfg.BASIC.CKPT_DIR, "best_score.pth")
    best_net_save = osp.join(cfg.BASIC.CKPT_DIR, "best_net.pth")
    best_prototoype_save = osp.join(cfg.BASIC.CKPT_DIR, "best_prototype.pth")

    # for diversity loss
    cen = get_center(prototype, opt.component)
    net = Mydata(cen)

    # optimizer
    params = [
        {'params': model.conv1.parameters(), 'lr': opt.found_lr / 10},
        {'params': model.bn1.parameters(), 'lr': opt.found_lr / 10},
        {'params': model.layer1.parameters(), 'lr': opt.found_lr / 8},
        {'params': model.layer2.parameters(), 'lr': opt.found_lr / 6},
        {'params': model.layer3.parameters(), 'lr': opt.found_lr / 4},
        {'params': model.layer4.parameters(), 'lr': opt.found_lr / 2}
    ]
    optimizer = optim.SGD(params=params, lr=opt.found_lr, momentum=opt.momentum)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=0.1)
    optimizer1 = torch.optim.Adam(params=net.parameters(), lr=opt.lr_prototype)
    scheduler1 = lr_scheduler.MultiStepLR(optimizer1, milestones=opt.milestones, gamma=0.1)

    # train
    print("start train feature")
    for iepoch in range(opt.epoch):
        print(f'\tepoch: {iepoch+1}')

        # save
        score_save = osp.join(cfg.BASIC.CKPT_DIR, "score_{}.pth".format(iepoch))

        # loss
        loss = train_prototype(model, train_loader, prototype, opt.component, optimizer, scheduler)
        print("net loss:{}".format(loss))
        if iepoch != 0:
            prototype = []
            fea_all = get_all_fea(model, train_loader, fea_all)
            for data in fea_all:
                centers, cluster_number = cluster_by_cosine(data, opt.component)
                prototype.append(centers)

            cen1 = ele_sum(temp_prototype, prototype, opt.component)
            prototype = update_center_by_dloss(prototype, cen1)
            cen = get_center(prototype, opt.component)
            net.update(cen)

        cen = net()
        optimizer1.zero_grad()
        prot = calculate_cls(cen)
        dloss = diversity_loss(prot)
        dloss.backward()
        optimizer1.step()
        scheduler1.step()
        print("dloss:{}".format(dloss))
        prototype = update_center_by_dloss(prototype, cen)

        predicts_val, val_label, val_ids = evaluate_by_prototype(model, val_loader, prototype, opt.component)
        predicts_test, test_label, test_ids = evaluate_by_prototype(model, test_loader, prototype, opt.component)
        lab_and_pre["val_scores"] = predicts_val
        lab_and_pre["test_scores"] = predicts_test
        lab_and_pre["val_targets"] = val_label
        lab_and_pre["test_targets"] = test_label
        lab_and_pre["val_ids"] = val_ids
        lab_and_pre["test_ids"] = test_ids


        temp_prototype = prototype

        torch.save(lab_and_pre, score_save)
        score, df = get_allresults_df(score_save)
        writer.add_scalar('score', score, iepoch+1)
        print(score)
        print(df)
        if score > max_score:
            max_score = score
            max_df = df
            torch.save(lab_and_pre, best_score_save)
            torch.save(model.state_dict(), best_net_save)
            torch.save(cen, best_prototoype_save)
        print(max_score)
    writer.close()
    print(max_df)
    print(max_score)


if __name__ == '__main__':
    main()
