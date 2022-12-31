import numpy as np
import torch.utils.data as data
from PIL import Image
import os
import torchvision.transforms as transforms
from lib.config.para import parse_opt
opt = parse_opt()

class Intentonomy(data.Dataset):

    def __init__(self, dir, anns, phase):
        self.dir = dir
        self.anns = anns
        # phase: train, val, test
        assert phase in ('train', 'val', 'test')
        self.phase = phase

    def __getitem__(self, item):
        pretrained_size = [224, 224]
        pretrained_means = [0.485, 0.456, 0.406]
        pretrained_stds = [0.229, 0.224, 0.225]
        img_trans_prototype = transforms.Compose([
            transforms.ToTensor(),
        ])
        if self.phase == 'train':
            data = self.anns[item]
            image_id = data['image_id']
            softprob = data['category_ids_softprob']
            img = Image.open(os.path.join(self.dir, image_id + '.jpg'))
            # transforms
            img_transforms = transforms.Compose([
                transforms.RandomRotation(opt.rotation),
                transforms.RandomHorizontalFlip(opt.HorizontalFlip),
                transforms.RandomCrop(pretrained_size, padding=opt.Crop_padding),
                # must be !
                transforms.ToTensor(),
                transforms.Normalize(mean=pretrained_means,std=pretrained_stds)
            ])
            img = img_transforms(img)

            softprob = np.array((softprob))
            label = softprob
            softprob = softprob / softprob.sum()
            return img, softprob, label

        elif self.phase == 'val':
            data = self.anns[item]
            image_id = data['image_id']
            label = data['category_ids']
            # for the same length
            label = np.pad(label,(0,opt.Intent_class-len(label)),mode='constant',constant_values=(-1,-1))
            img = Image.open(os.path.join(self.dir, image_id + '.jpg'))
            img_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=pretrained_means, std=pretrained_stds)
            ])
            img = img_transforms(img)
            return img, label, image_id
        else:
            data = self.anns[item]
            if "image_id" in data:
                image_id = data["image_id"]
                label = data['category_ids']
            if "image_category_id" in data:
                image_id = data['image_category_id']
                label = data["category_category_ids"]
            label = np.pad(label, (0, opt.Intent_class - len(label)), mode='constant', constant_values=(-1, -1))
            img = Image.open(os.path.join(self.dir, image_id + '.jpg'))
            img_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=pretrained_means, std=pretrained_stds)
            ])
            img = img_transforms(img)
            return img, label, image_id


    def __len__(self):
        return len(self.anns)

if __name__ == '__main__':
    import json
    import torch
    dir = '/disk/yk/Intentonomy/dataset/image/'
    file = '/disk/yk/Intentonomy/dataset/annotation/intentonomy_train2020.json'

    with open(file, 'r') as f:
        datas = json.load(f)
    annotations = datas['annotations']

    train_fun = Intentonomy(dir, annotations, phase='train')
    train_loader = torch.utils.data.DataLoader(train_fun, batch_size=2, shuffle=True)

    for img, softprob in train_loader:
        print(type(img), img.size(), type(softprob), softprob.size())
        img_tmp = img[0, 0, :, :]
        img_tmp = img_tmp.numpy()
        sl = softprob.numpy()


