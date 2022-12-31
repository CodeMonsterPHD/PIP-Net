import torch.nn.functional as F
from lib.utils.utils import get_center
import torch
import numpy as np
from lib.config.para import parse_opt
opt = parse_opt()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_prototype(model,loader,prototype,component,optimizer,scheduler):
    c = get_center(prototype,component).detach()   # [component * Intent_class , 2048]
    loss_log = 0
    model.train()
    model.get_center(c)
    for i, (img, softlabel, _) in enumerate(loader):
        img = img.to(device)
        softlabel = softlabel.to(device)
        optimizer.zero_grad()
        pre = model(img,'train')
        loss = -torch.mean(torch.sum(softlabel * F.log_softmax(pre, dim=1), dim=1), dim=0)
        loss.backward()
        optimizer.step()
        loss_log = loss_log + loss.item()
    loss_avg = loss_log / len(loader)
    scheduler.step()

    return loss_avg


def evaluate_by_prototype(model, loader,prototype,component):
    c = get_center(prototype,component)
    model.eval()
    model.get_center(c)
    predcts = np.zeros((len(loader.dataset),opt.Intent_class))
    softmax = torch.nn.Softmax(dim=1)
    label_all = []
    img_ids = []
    count = 0
    for i, (img,label,img_id) in enumerate(loader):
        img = img.to(device)
        label = label.to(device)
        img_ids.append(img_id)
        with torch.no_grad():
            prediction = model(img,'test')
            prediction = softmax(prediction)
            prediction = prediction.cpu().numpy()
            label = label.cpu().numpy()
            # label style: [7,16,-1,-1,-1,...,]   -->   [7,16]
            for i in range(len(prediction)):
                predcts[count] = prediction[i]
                list_tmp = []
                j = label[i]
                for i in j:
                    if i != -1:
                        list_tmp.append(i)
                label_all.append(list_tmp)
                count = count+1

    return predcts, label_all, img_ids